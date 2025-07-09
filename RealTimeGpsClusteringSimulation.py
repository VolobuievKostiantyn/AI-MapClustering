import random
import time
import folium
from sklearn.cluster import KMeans
import numpy as np

# AI-MapClustering.
# This script does the following
# : Simulates incoming GPS data in real time.
# : Uses K-Means clustering on the data.
# : Plots clusters on a static map using folium (since Google Maps SDK is not used in Python).
 
# Step 1: Simulate live GPS data stream
def generate_random_gps(center, radius_km=1):
    """Generate a random point near a center (lat, lon)"""
    radius_deg = radius_km / 111  # approx conversion
    lat = center[0] + random.uniform(-radius_deg, radius_deg)
    lon = center[1] + random.uniform(-radius_deg, radius_deg)
    return (lat, lon)

# Step 2: Simulated incoming GPS points
def simulate_gps_data(n=50, center=(37.7750, -122.4190)):
    return [generate_random_gps(center) for _ in range(n)]

# Step 3: Perform KMeans clustering
def cluster_gps_data(points, k=3):
    coords = np.array(points)
    kmeans = KMeans(n_clusters=k, n_init='auto')
    labels = kmeans.fit_predict(coords)
    return labels, kmeans.cluster_centers_

# Step 4: Plot on map using folium
def plot_clusters(points, labels, centers, map_center):
    m = folium.Map(location=map_center, zoom_start=13)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for point, label in zip(points, labels):
        folium.CircleMarker(
            location=point,
            radius=5,
            color=colors[label % len(colors)],
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
        
    for center in centers:
        folium.Marker(
            location=center.tolist(),
            icon=folium.Icon(color='black', icon='crosshairs')
        ).add_to(m)
        
    return m

# Main simulation
if __name__ == "__main__":
    gps_points = simulate_gps_data(n=100)
    labels, centers = cluster_gps_data(gps_points, k=3)
    map_center = (np.mean([p[0] for p in gps_points]), np.mean([p[1] for p in gps_points]))
    
    gps_map = plot_clusters(gps_points, labels, centers, map_center)
    
    # Save to HTML to visualize
    gps_map.save("clustered_map.html")
    print("Clustered map saved to clustered_map.html")
