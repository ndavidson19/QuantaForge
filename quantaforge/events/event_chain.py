import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class EnhancedEventChain:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_event(self, event: Event, parent_events: List[Tuple[Event, float]] = None):
        self.graph.add_node(event)
        if parent_events:
            for parent_event, probability in parent_events:
                self.graph.add_edge(parent_event, event, probability=probability)

    def get_event_probability(self, event: Event) -> float:
        if self.graph.in_degree(event) == 0:
            return 1.0
        
        total_prob = 0
        for parent in self.graph.predecessors(event):
            parent_prob = self.get_event_probability(parent)
            edge_prob = self.graph[parent][event]['probability']
            total_prob += parent_prob * edge_prob
        
        return total_prob

    def resolve_chain(self, event_manager: EventManager):
        for event in nx.topological_sort(self.graph):
            if np.random.random() < self.get_event_probability(event):
                event_manager.add_event(event)

    def visualize(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.graph, 'probability')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Event Chain Visualization")
        plt.axis('off')
        plt.show()

class EventChainManager:
    def __init__(self):
        self.chains: Dict[str, EnhancedEventChain] = {}

    def create_chain(self, chain_id: str) -> EnhancedEventChain:
        chain = EnhancedEventChain()
        self.chains[chain_id] = chain
        return chain

    def get_chain(self, chain_id: str) -> EnhancedEventChain:
        return self.chains.get(chain_id)

    def resolve_all_chains(self, event_manager: EventManager):
        for chain in self.chains.values():
            chain.resolve_chain(event_manager)

    def visualize_chain(self, chain_id: str):
        chain = self.get_chain(chain_id)
        if chain:
            chain.visualize()
        else:
            print(f"Chain with id {chain_id} not found.")