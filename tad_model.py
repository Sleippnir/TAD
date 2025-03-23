Import random
import numpy as np  # Import NumPy
from mesa import Agent, Model
from mesa.time import RandomActivation
#from mesa.datacollection import DataCollector # Removed
import h5py  # Import h5py
import json

# --- Voter Agent Class ---

class VoterAgent(Agent):
    """
    A voter agent in the TAD model.  Represents individual voters with
    varying attributes and behaviors related to information processing,
    opinion formation, and voting.
    """
    def __init__(self, unique_id, model, information_level, policy_preferences,
                 susceptibility_to_misinformation, engagement_level, party_support):
        super().__init__(unique_id, model)
        self.information_level = information_level
        self.policy_preferences = policy_preferences  # Dictionary {category: value}
        self.susceptibility_to_misinformation = susceptibility_to_misinformation
        self.engagement_level = engagement_level
        self.party_support = party_support  # Dictionary {party_id: support_value}
        self.belief_state = {}  # {policy_id: belief_value}
        self.policy_opinions = {}  # {policy_id: {category: opinion_value}}
        self.voted_for = None  # Keep track of last voted party
        self.openness = random.random()
        self.opinions = {} # Overall opinion, average over categories


    def receive_information(self, policy):
        """
        Receives information about a policy and processes it, including
        checking for misinformation and updating understanding.

        Args:
            policy (dict): The policy being considered, represented as a dictionary.
                           Must have keys "unique_id", "clarity_score", and "ratings".
        """
        misinformation_encountered = False
        policy_ratings = policy["ratings"]
        clarity_score = policy["clarity_score"]


        if random.random() < self.model.global_misinformation_level:
            misinformation_encountered = True
            clarity_score = clarity_score - random.uniform(0.1, 0.3)
            if clarity_score < 0:
                clarity_score = 0

            misinformation_slant = random.choice([-1, 1])
            distorted_ratings = {}
            for category, rating in policy_ratings.items():
                distorted_ratings[category] = rating + (misinformation_slant * random.uniform(0, self.model.distortion_factor))
                if distorted_ratings[category] > 1:  distorted_ratings[category] = 1
                if distorted_ratings[category] < -1: distorted_ratings[category] = -1
            policy_ratings = distorted_ratings

        understanding = self.information_level * clarity_score + random.gauss(0, 0.1)
        if understanding > 1: understanding = 1
        if understanding < 0: understanding = 0

        pass_test = understanding > self.model.comprehension_threshold

        self.update_belief_state(policy, pass_test, misinformation_encountered)
        self.update_opinion(policy, policy_ratings, pass_test)
        self.update_information_level(pass_test, clarity_score)
        self.update_engagement_level(pass_test, misinformation_encountered)


    def update_belief_state(self, policy, pass_test, misinformation_encountered):
        """Updates the agent's belief state regarding a policy."""

        if policy["unique_id"] not in self.belief_state:
            self.belief_state[policy["unique_id"]] = 0

        if pass_test:
            belief_change = 1
        else:
            if misinformation_encountered:
                belief_change = -1 * self.susceptibility_to_misinformation
            else:
                belief_change = 0

        self.belief_state[policy["unique_id"]] += self.openness * self.model.belief_update_rate * belief_change

         # Cap the values.
        if self.belief_state[policy["unique_id"]] > 1:  self.belief_state[policy["unique_id"]] = 1
        if self.belief_state[policy["unique_id"]] < -1: self.belief_state[policy["unique_id"]] = -1


    def update_opinion(self, policy, policy_ratings, pass_test):
        """Calculates and updates the agent's opinion of a policy, per category."""

        if policy["unique_id"] not in self.policy_opinions:
            self.policy_opinions[policy["unique_id"]] = {}

        for category, rating in policy_ratings.items():
            policy_score = 1 - abs(self.policy_preferences[category] - rating)
            if policy_score < 0:
                policy_score = 0

            if category not in self.policy_opinions[policy["unique_id"]]:
                self.policy_opinions[policy["unique_id"]][category] = 0

            if pass_test:
                comprehension_weight = 0.8
            else:
                comprehension_weight = 0.2

            self.policy_opinions[policy["unique_id"]][category] = (
                self.policy_opinions[policy["unique_id"]][category] * (1 - comprehension_weight)
            ) + (policy_score * comprehension_weight)

        # Calculate and store OVERALL opinion (average of category opinions)
        if policy["unique_id"] in self.policy_opinions: #Should always be true
            self.opinions[policy["unique_id"]] = sum(self.policy_opinions[policy["unique_id"]].values()) / len(self.policy_opinions[policy["unique_id"]])


    def update_information_level(self, pass_test, clarity_score):
        """Updates the agent's information level based on test results and clarity."""
        if pass_test:
            self.information_level += 0.1 * clarity_score * (1 - self.information_level)
        else:
            self.information_level -= 0.05 * clarity_score * self.information_level
        if self.information_level > 1: self.information_level = 1
        if self.information_level < 0: self.information_level = 0


    def update_engagement_level(self, pass_test, misinformation_encountered):
        """Updates the agent's engagement level based on test results and misinformation."""
        if pass_test:
            self.engagement_level += 0.05 * (1 - self.engagement_level)
        elif misinformation_encountered:
            self.engagement_level -= 0.05 * self.engagement_level
        if self.engagement_level > 1: self.engagement_level = 1
        if self.engagement_level < 0: self.engagement_level = 0


    def participate(self):
        """Decides whether the voter participates in comprehension testing (simplified)."""
        return random.random() < self.engagement_level


    def calculate_party_utility(self, party):
        """Calculates the agent's utility for a given party."""

        total_opinion = 0
        num_policies = 0
        for policy_id in party.policies_supported:
            if policy_id in self.policy_opinions:
                # overall_policy_opinion = sum(self.policy_opinions[policy_id].values()) / len(self.policy_opinions[policy_id]) # REPLACED
                overall_policy_opinion = self.opinions[policy_id] # Use pre-calculated overall opinion
                total_opinion += overall_policy_opinion
                num_policies += 1

        if num_policies > 0:
            average_policy_opinion = total_opinion / num_policies
        else:
            average_policy_opinion = 0.5

        utility = (self.model.policy_alignment_weight * average_policy_opinion) + \
                  (self.model.competence_weight * party.competence)
        if self.voted_for == party.unique_id:
             utility += self.model.loyalty_bonus
        return utility


    def vote(self):
        """Decides which party to vote for."""

        best_party = None
        highest_utility = -1

        for party in self.model.schedule.agents_by_type[PartyAgent]:
            utility = self.calculate_party_utility(party)
            if utility > highest_utility:
                highest_utility = utility
                best_party = party.unique_id
        self.voted_for = best_party
        return best_party

# --- Party Agent Class (Simplified) ---
class PartyAgent(Agent):
    """A political party agent."""
    def __init__(self, unique_id, model, ideology):
        super().__init__(unique_id, model)
        self.ideology = ideology  # A dictionary like policy_preferences
        self.policies_supported = []  # List of policy IDs
        self.competence = random.uniform(0.4, 0.9) #More realistic competence
        self.seats = 0 # Initialize seats

    def step(self):
        # Only consider supporting the policy if it's new
        if self.model.current_policy and self.model.current_policy["unique_id"] not in self.policies_supported:
            ideology_alignment = 0
            for category, rating in self.model.current_policy["ratings"].items():
                # Calculate alignment with party's ideology
                ideology_alignment += 1 - abs(self.ideology[category] - rating)

            average_alignment = ideology_alignment / len(self.model.current_policy["ratings"])

            # Support the policy based on alignment and a threshold
            if average_alignment > self.model.party_support_threshold:
                self.policies_supported.append(self.model.current_policy["unique_id"])

# --- Government Agent Class (Simplified) ---
class GovernmentAgent(Agent):
    """An agent representing the government, responsible for proposing policies."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.policy_id_counter = 0

    def create_policy(self):
        """Creates a new policy with random ratings."""
        self.policy_id_counter += 1
        policy_id = self.policy_id_counter
        clarity_score = random.normalvariate(0.7, 0.15)
        if clarity_score > 1: clarity_score = 1
        if clarity_score < 0: clarity_score = 0

        policy_ratings = {}
        for category in self.model.policy_categories:
            policy_ratings[category] = random.uniform(-1, 1)

        return {
            "unique_id": policy_id,
            "clarity_score": clarity_score,
            "ratings": policy_ratings,
        }

    def step(self):
      if self.schedule.steps % self.model.policy_creation_frequency == 0: #Only create policies sometimes
        # Create and store the new policy
        self.model.current_policy = self.create_policy()

# --- Policy Class ---
# class Policy(Agent): # REMOVED - Policy is no longer an agent
#     """Represents a policy proposal."""
#     def __init__(self, unique_id, clarity_score, ratings):
#         super().__init__(unique_id, model)
#         self.clarity_score = clarity_score
#         self.ratings = ratings  # Dictionary {category: rating}

# --- Model Class ---
from mesa import Model
from mesa.time import RandomActivation
#from mesa.datacollection import DataCollector #Removed
import h5py
import json

class TADModel(Model):
    """
    The main model for the Transparent and Accountable Democracy simulation.
    """
    def __init__(self, num_voters=2000, comprehension_threshold=0.6,
                 belief_update_rate=0.2, adjustment_rate=0.01,
                 distortion_factor=0.3, policy_alignment_weight=0.7,
                 competence_weight=0.2, loyalty_bonus=0.1,
                 global_misinformation_level=0.25, policy_creation_frequency = 5,
                 election_interval=24, party_support_threshold = 0.6, num_seats=51,
                 representation_threshold = 0.03, seed=None): # Added parameters
        super().__init__()
        self.num_voters = num_voters
        self.global_misinformation_level = global_misinformation_level
        self.comprehension_threshold = comprehension_threshold
        self.belief_update_rate = belief_update_rate
        self.adjustment_rate = adjustment_rate
        self.distortion_factor = distortion_factor
        self.policy_alignment_weight = policy_alignment_weight
        self.competence_weight = competence_weight
        self.loyalty_bonus = loyalty_bonus
        self.policy_categories = ["Economic Security", "Healthcare", "Education", "Environment", "Social Justice"]
        self.schedule = RandomActivation(self)
        #self.gov_schedule = RandomActivation(self) # No longer needed
        self.current_policy = None
        #self.current_policy_ratings = None # No longer needed
        self.policy_creation_frequency = policy_creation_frequency
        self.election_interval = election_interval
        self.steps_since_last_election = 0
        self.party_support_threshold = party_support_threshold
        self.num_seats = num_seats  # Number of seats in parliament
        self.representation_threshold = representation_threshold #Minimum percentage to get a seat.
        self.running = True #Mesa needs this.
        self.data_file = None

        # Set the seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # Create agents
        for i in range(self.num_voters):
            information_level = random.random()
            policy_preferences = {}
            for cat in self.policy_categories:
                policy_preference[cat] = random.normalvariate(0, 0.33)
                if policy_preference[cat] > 1: policy_preference[cat] = 1
                if policy_preference[cat] < -1: policy_preference[cat] = -1

            susceptibility_to_misinformation = random.random()
            engagement_level = random.random()
            party_support = {}
            if policy_preference["Economic Security"] < -0.33: # Using one as example
                party_support["CL"] = random.uniform(0.5, 1)
                party_support["C"] = random.uniform(0, 0.4)
                party_support["CR"] = random.uniform(0, 0.1)
            elif policy_preference["Economic Security"] > 0.33:
                party_support["CL"] = random.uniform(0, 0.1)
                party_support["C"] = random.uniform(0, 0.4)
                party_support["CR"] = random.uniform(0.5, 1)
            else:
                party_support["CL"] = random.uniform(0, 0.4)
                party_support["C"] = random.uniform(0.5, 1)
                party_support["CR"] = random.uniform(0, 0.4)

            total_support = sum(party_support.values())
            for party in party_support:
                party_support[party] /= total_support


            a = VoterAgent(i, self, information_level, policy_preferences,
                            susceptibility_to_misinformation, engagement_level,
                            party_support)
            self.schedule.add(a)

        #Add Parties
        party_CL = PartyAgent("CL", self, {
            "Economic Security": -0.8,
            "Healthcare": -0.7,
            "Education": -0.6,
            "Environment": -0.5,
"Social Justice": -0.9
        })
        party_C = PartyAgent("C", self, {
            "Economic Security": 0.0,
            "Healthcare": 0.1,
            "Education": 0.0,
            "Environment": -0.1,
            "Social Justice": 0.0
        })
        party_CR = PartyAgent("CR", self, {
            "Economic Security": 0.8,
            "Healthcare": 0.7,
            "Education": 0.6,
            "Environment": 0.5,
            "Social Justice": 0.9
        })
        self.schedule.add(party_CL)
        self.schedule.add(party_C)
        self.schedule.add(party_CR)

        #Add goverment
        goverment = GovernmentAgent("goverment", self)
        self.schedule.add(goverment)

        # --- Data Collection Setup (HDF5) ---
        self.setup_data_collection()

    def setup_data_collection(self):
        """Sets up HDF5 file and datasets."""
        self.data_file = h5py.File("simulation_data.hdf5", "w") #Create hdf5 file

        # Create datasets:
        self.model_dataset = self.data_file.create_dataset(
            "model_data",
            (0, 6),  # Initial shape: (0 rows, 6 columns) - will grow dynamically
            maxshape=(None, 6),  # Unlimited rows, fixed columns
            dtype=float,
            chunks=(100, 6) # Write in chunks
        )
        self.agent_dataset = self.data_file.create_dataset(
            "agent_data",
            (0, 6), # Initial shape: (0 rows, 6 columns) - will grow dynamically.
            maxshape=(None, 6), # Unlimited rows, fixed columns
            dtype=float,
            chunks=(1000, 6) # Write in chunks
        )

        # Add attributes (metadata):
        self.model_dataset.attrs["description"] = "Model-level data for TAD simulation"
        self.agent_dataset.attrs["description"] = "Agent-level data for TAD simulation"

        # Add parameters as attributes to the model dataset:
        self.metadata = {
            "num_voters": self.num_voters,
            "comprehension_threshold": self.comprehension_threshold,
            "belief_update_rate": self.belief_update_rate,
            "adjustment_rate": self.adjustment_rate,
            "distortion_factor": self.distortion_factor,
            "policy_alignment_weight": self.policy_alignment_weight,
            "competence_weight": self.competence_weight,
            "loyalty_bonus": self.loyalty_bonus,
            "initial_misinformation_level": self.global_misinformation_level,
            "policy_creation_frequency": self.policy_creation_frequency,
            "election_interval": self.election_interval,
            "party_support_threshold": self.party_support_threshold,
            "num_seats": self.num_seats,
            "representation_threshold": self.representation_threshold,
            "seed": self.random.seed #Using the function to obtain the seed
        }

        for param_name, param_value in self.metadata.items():
            self.model_dataset.attrs[param_name] = param_value



    def collect_data(self):
        """Collects data from the model and agents and appends it to the HDF5 datasets."""

        # Model-level data:
        model_data = [
            self.schedule.steps,
            self.global_misinformation_level,
            1 if self.steps_since_last_election == 0 else 0,  # Election indicator
            self.get_seat_counts().get("CL", 0),
            self.get_seat_counts().get("C", 0),
            self.get_seat_counts().get("CR", 0)
        ]

        # Append model data to the model dataset, resizing if necessary
        self.model_dataset.resize((self.model_dataset.shape[0] + 1), axis=0)
        self.model_dataset[-1] = model_data


        # Agent-level data:
        agent_data_list = [] # We create a list to append agents
        for agent in self.schedule.agents_by_type[VoterAgent]:
            agent_data = [
                self.schedule.steps, # Step
                agent.unique_id,
                1,  # Agent type: 1 for VoterAgent
                agent.information_level,
                agent.engagement_level,
                agent.voted_for if agent.voted_for is not None else -1 # voted_for can be none
            ]
            agent_data_list.append(agent_data)

        for agent in self.schedule.agents_by_type[PartyAgent]:
            agent_data = [
                self.schedule.steps, # Step
                agent.unique_id,
                2,  # Agent type: 2 for PartyAgent
                agent.ideology,
                agent.competence,
                agent.seats
            ]
            agent_data_list.append(agent_data)
        #Append to hdf5
        new_rows = len(agent_data_list)
        self.agent_dataset.resize((self.agent_dataset.shape[0] + new_rows), axis=0)
        self.agent_dataset[-new_rows:] = agent_data_list

    def step(self):
        """Advance the model by one step."""

        # The government creates a policy each *policy_creation_frequency* steps
        if self.schedule.steps % self.policy_creation_frequency == 0:
                self.current_policy = self.schedule.agents_by_type[GovernmentAgent][0].create_policy()

        self.schedule.step()  # Activate all agents
        self.update_global_misinformation()
        if self.schedule.steps % 50 == 0: # Collect data every 50 steps
            self.collect_data()
        self.steps_since_last_election += 1

        # Check for election
        if self.steps_since_last_election >= self.election_interval:
            self.hold_election()
            self.steps_since_last_election = 0

    def get_vote_counts(self):
        """Helper function to count votes for each party."""
        vote_counts = {}
        for party in self.schedule.agents_by_type[PartyAgent]:
            vote_counts[party.unique_id] = 0
        for agent in self.schedule.agents_by_type[VoterAgent]:
            if agent.voted_for:  # Check if the agent voted
                vote_counts[agent.voted_for] += 1
        return vote_counts

    def get_seat_counts(self):
        """Helper function to retrieve seat counts for each party (after an election)."""
        seat_counts = {}
        for party in self.schedule.agents_by_type[PartyAgent]:
            seat_counts[party.unique_id] = party.seats if hasattr(party, 'seats') else 0
        return seat_counts

    def update_global_misinformation(self):
        """Updates the Global_Misinformation_Level based on agents' average belief."""
        total_belief = 0
        agent_count = 0
        for agent in self.schedule.agents_by_type[VoterAgent]: #Access the agents by type
            for belief in agent.belief_state.values():
                total_belief += belief
                agent_count +=1

        if agent_count > 0:
            average_belief = total_belief / agent_count
        else:
            average_belief = 0

        # Map belief to a value between 0 and 1
        average_belief_factor = (average_belief + 1) / 2

        self.global_misinformation_level = (self.global_misinformation_level * (1 - self.model.adjustment_rate)) + (average_belief_factor * self.model.adjustment_rate)
        if self.global_misinformation_level < 0 : self.global_misinformation_level = 0
        if self.global_misinformation_level > 1 : self.global_misinformation_level = 1


    def hold_election(self):
        """Holds an election using the D'Hondt method and updates party competence."""

        # 1. Get Vote Counts:
        vote_counts = self.get_vote_counts()

        # 2. Apply Representation Threshold:
        eligible_parties = {}  # Parties that meet the threshold
        total_votes = sum(vote_counts.values())
        for party_id, votes in vote_counts.items():
            if total_votes > 0: #Avoid ZeroDivisionError
                vote_share = votes / total_votes
                if vote_share >= self.representation_threshold:
                    eligible_parties[party_id] = votes
            else:
                eligible_parties[party_id] = 0 #If there are no votes, no party gets in.

        # 3. D'Hondt Method Implementation:
        seats = {}  # {party_id: num_seats}
        for party in eligible_parties:
            seats[party] = 0
        quotients = {} # { (party_id, divisor) : quotient }

        #Calculate quotients
        for party_id, votes in eligible_parties.items():
            for divisor in range(1, self.num_seats + 1):
                quotients[(party_id, divisor)] = votes / divisor

        #Allocate seats
        for _ in range(self.num_seats):
            if quotients:  # Check if quotients is not empty
                winner = max(quotients, key=quotients.get)  # Find the highest quotient
                party_id = winner[0]
                seats[party_id] += 1
                del quotients[winner]  # Remove the winning quotient

        # 4. Update Party Competence (Simple Example):
        #    (This is a placeholder.  A more sophisticated model would be needed.)
        for party in self.schedule.agents_by_type[PartyAgent]:
            if party.unique_id in seats:
                party.seats = seats[party.unique_id]  # Assign the correct number of seats.
                party.competence = min(1,max(0,party.competence + 0.05 * (seats[party.unique_id]/self.num_seats))) # Get up to .05 competence by seat
            else:
                party.seats = 0 # Ensure the party has 0 seats.
                party.competence = min(1,max(0,party.competence - 0.05)) #Reduce if no seats are won.
            #Further logic for losing competence by not passing policy, etc.
        # 5. Reset Voter's voted_for attribute
        for agent in self.schedule.agents_by_type[VoterAgent]:
            agent.voted_for = None

        # 6. Output Results (and Store for Data Collection):
        print(f"Election Results: {seats}")  # Basic output
        #self.datacollector.collect(self) #Removed data collector
        #self.collect_data() # Handled in step.

        # 7. (Later) Government Formation Logic would go here.

    def __del__(self):
        """ Closes the file when the simulation ends."""
        if self.data_file:
            self.data_file.close()
