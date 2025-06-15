import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

def mark_multiple_countries_with_distances(actual_coordinates, country_codes):
    """
    Marks multiple countries on a world map and displays the distances between:
    1. Geometric centers of countries based on their ISO Alpha-2 codes.
    2. Actual coordinates provided as input.

    Args:
        actual_coordinates (tuple): A tuple of latitude and longitude (e.g., (52.2297, 21.0122)).
        country_codes (list): A list of ISO Alpha-2 codes of countries (e.g., ['PL', 'JP']).

    Returns:
        folium.Map: A folium map object with the marked locations and distances.
    """
    country_codes = [country_code.upper() for country_code in country_codes]

    # Initialize geolocator
    geolocator = Nominatim(user_agent="geoapi")

    # Create a folium map centered at the first actual coordinate
    world_map = folium.Map(location=actual_coordinates, zoom_start=4)

    # Add marker for the actual coordinates
    folium.Marker(
        location=actual_coordinates,
        popup="Actual Coordinates",
        icon=folium.Icon(color="red", icon="map-marker")
    ).add_to(world_map)

    # Initialize a list to store distances
    distances = []

    # Iterate over the list of country codes
    for country_code in country_codes:
        # Get geometric center of the country
        location = geolocator.geocode(country_code)
        if not location:
            print(f"Could not find geometric center for country code: {country_code}")
            continue
        country_center = (location.latitude, location.longitude)

        # Calculate the distance between the two points
        distance = geodesic(country_center, actual_coordinates).kilometers
        distances.append((country_code, distance))

        # Add marker for the geometric center
        folium.Marker(
            location=country_center,
            popup=f"Geometric Center of {country_code.upper()}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(world_map)

    # Add distance display in the top-right corner
    distance_html = "<div style='position: fixed; top: 10px; right: 10px; background-color: white; padding: 10px; border: 1px solid black; z-index: 1000;'>"
    for country_code, distance in distances:
        distance_html += f"Distance to {country_code.upper()}: {distance:.2f} km<br>"
    distance_html += "</div>"
    world_map.get_root().html.add_child(folium.Element(distance_html))

    return world_map
