import os
import sys
import json
import numpy as np
import networkx as nx

# Add workspace to path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from custom_environment.power_grid.csprl_adapter import create_adapter_for_location
import custom_environment.helpers as H

def preprocess_scaling(location="DongDa"):
    print(f"--- Preprocessing scaling constants for {location} ---")
    
    # Paths
    graph_dir = os.path.join(base_dir, "custom_environment", "data", "Graph", location)
    graph_file = os.path.join(graph_dir, f"{location}.graphml")
    node_file = os.path.join(graph_dir, f"nodes_extended_{location}.txt")
    
    if not os.path.exists(node_file):
        print(f"Error: Node file not found at {node_file}")
        return
        
    # Load graph and nodes
    graph, node_list = H.prepare_graph(graph_file, node_file)
    
    # Extract raw node attributes
    x_coords = []
    y_coords = []
    land_prices = []
    demands = []
    street_counts = []
    
    for _, attrs in node_list:
        if 'x' in attrs: x_coords.append(attrs['x'])
        if 'y' in attrs: y_coords.append(attrs['y'])
        if 'land_price' in attrs: land_prices.append(attrs['land_price'])
        if 'demand' in attrs: demands.append(attrs['demand'])
        if 'street_count' in attrs: street_counts.append(attrs['street_count'])
        
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    land_prices = np.array(land_prices)
    demands = np.array(demands)
    street_counts = np.array(street_counts)
    
    # Load grid adapter to get grid features
    try:
        adapter = create_adapter_for_location(location, base_path=os.path.join(base_dir, "custom_environment", "data"))
        extended_nodes = adapter.extend_node_features(node_list, [])
        grid_distances = [n[1].get('grid_distance_km', 3.0) for n in extended_nodes]
        # Replace inf with nan and drop
        grid_distances = [d for d in grid_distances if d != float('inf') and not np.isinf(d)]
        grid_distances = np.array(grid_distances) if grid_distances else np.array([1.21]) # fallback
        
        bus_capacities = adapter.get_all_bus_capacities([])
        bus_capacities = np.array(bus_capacities) if len(bus_capacities) > 0 else np.array([10.0]) # fallback
    except Exception as e:
        print(f"Warning: Grid adapter failed: {e}. Using fallback statistics.")
        grid_distances = np.array([1.21])
        bus_capacities = np.array([10.0])
        
    # Get edge lengths
    edge_lengths = []
    for _, _, data in graph.edges(data=True):
        if 'length' in data:
            edge_lengths.append(data['length'])
    edge_lengths = np.array(edge_lengths) if edge_lengths else np.array([100.0])
    
    # Compute robust scaling boundaries (using 95th percentiles to avoid outlier compression)
    scaling_data = {
        "location": location,
        "x_min": float(x_coords.min()) if len(x_coords) > 0 else 105.798,
        "x_max": float(x_coords.max()) if len(x_coords) > 0 else 105.843,
        "y_min": float(y_coords.min()) if len(y_coords) > 0 else 20.997,
        "y_max": float(y_coords.max()) if len(y_coords) > 0 else 21.032,
        
        # 95th percentile for heavy-tailed or skewed features
        "land_price_max": float(np.percentile(land_prices, 95)) if len(land_prices) > 0 else 186.2,
        "demand_max": float(np.percentile(demands, 95)) if len(demands) > 0 else 0.91,
        "grid_dist_max": float(np.percentile(grid_distances, 95)) if len(grid_distances) > 0 else 1.21,
        "grid_mw_max": float(np.percentile(bus_capacities, 95)) if len(bus_capacities) > 0 else 12.8,
        "street_count_max": float(street_counts.max()) if len(street_counts) > 0 else 5.0,
        "road_length_max": float(np.percentile(edge_lengths, 95)) if len(edge_lengths) > 0 else 1000.0,
        
        # Target constraints and typical operational limits
        "dist_to_station_max": 2.0, # Max local distance threshold for station coverage in km
        "capability_max": 1.5,      # Realistic maximum capability in MW
        "benefit_max": 2.0          # Realistic maximum coverage benefit score
    }
    
    output_file = os.path.join(graph_dir, "scaling_constants.json")
    with open(output_file, "w") as f:
        json.dump(scaling_data, f, indent=2)
        
    print(f"Saved scaling constants to {output_file}:")
    print(json.dumps(scaling_data, indent=2))

if __name__ == "__main__":
    loc = "DongDa"
    if len(sys.argv) > 1:
        loc = sys.argv[1]
    preprocess_scaling(loc)
