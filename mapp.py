import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium import plugins
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import random

# Pour l'IA (Optionnel)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    pass

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="Casa Logistics Brain", page_icon="🚛")

# --- CONSTANTES GLOBALES (CORRECTION DU BUG) ---
# On définit le centre ici pour qu'il soit toujours connu
CENTER_LAT = 33.5890
CENTER_LON = -7.6250

# CSS pour le Design
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    div.stButton > button {
        background: linear-gradient(45deg, #4285F4, #0F9D58);
        color: white; border: none; border-radius: 8px;
        padding: 10px 20px; font-weight: bold; transition: 0.3s;
    }
    div.stButton > button:hover { transform: scale(1.05); }
    .metric-card {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. MOTEUR CARTOGRAPHIQUE (CACHE) ---
@st.cache_resource
def load_graph():
    """Charge le graphe de Casablanca et le garde en mémoire cache."""
    print("⏳ Chargement du graphe OSM...")
    center = (CENTER_LAT, CENTER_LON)
    # Rayon de 5km
    G = ox.graph_from_point(center, dist=5000, network_type='drive')
    
    # Ajout des vitesses
    hwy_speeds = {'residential': 30, 'secondary': 50, 'tertiary': 40, 'primary': 60, 'trunk': 80}
    G = ox.add_edge_speeds(G, hwy_speeds=hwy_speeds)
    G = ox.add_edge_travel_times(G)
    return G

try:
    G = load_graph()
except Exception as e:
    st.error(f"Erreur de chargement de la carte : {e}")
    st.stop()

# --- 2. GÉNÉRATION DE DONNÉES (100 POINTS) ---
if 'df_points' not in st.session_state:
    n_points = 100
    data = []
    for i in range(n_points):
        lat = CENTER_LAT + random.uniform(-0.04, 0.04)
        lon = CENTER_LON + random.uniform(-0.05, 0.05)
        status = random.choice(['Pending', 'Delivered', 'Urgent', 'Failed'])
        priority = random.choice([1, 2, 3])
        
        data.append({
            'id': i,
            'lat': lat,
            'lon': lon,
            'status': status,
            'priority': priority,
            'name': f"Client #{i+100}"
        })
    st.session_state.df_points = pd.DataFrame(data)

df = st.session_state.df_points

# --- 3. ALGORITHME D'OPTIMISATION ---
def optimize_route(graph, points_df):
    if points_df.empty:
        return [], 0, []

    coords = points_df[['lat', 'lon']].values
    
    # Map Matching
    nodes = ox.distance.nearest_nodes(graph, coords[:, 1], coords[:, 0])
    points_df = points_df.copy()
    points_df['node'] = nodes

    # TSP Heuristique (Nearest Neighbor)
    path_indices = [0]
    unvisited = set(range(1, len(points_df)))
    curr_idx = 0

    while unvisited:
        current_pos = coords[curr_idx].reshape(1, -1)
        remaining_indices = list(unvisited)
        remaining_pos = coords[remaining_indices]
        
        dists = cdist(current_pos, remaining_pos, metric='euclidean')
        nearest_local_idx = np.argmin(dists)
        nearest_global_idx = remaining_indices[nearest_local_idx]
        
        path_indices.append(nearest_global_idx)
        unvisited.remove(nearest_global_idx)
        curr_idx = nearest_global_idx

    # Reconstruction route réelle
    full_route_geometry = []
    total_distance_m = 0
    ordered_nodes = points_df.iloc[path_indices]['node'].tolist()
    
    for i in range(len(ordered_nodes) - 1):
        try:
            u = ordered_nodes[i]
            v = ordered_nodes[i+1]
            route = nx.shortest_path(graph, u, v, weight='travel_time')
            dist = nx.path_weight(graph, route, weight='length')
            total_distance_m += dist
            for node in route:
                full_route_geometry.append((graph.nodes[node]['y'], graph.nodes[node]['x']))
        except:
            continue
            
    return full_route_geometry, total_distance_m, path_indices

# --- 4. AGENT IA ---
def ask_agent(question, context_data):
    api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        return "⚠️ Entrez une clé API OpenAI."
    try:
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
        template = "Tu es un expert logistique. Données: {context}. Question: {question}. Réponds court."
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        summary = context_data['status'].value_counts().to_dict()
        return chain.invoke({"context": str(summary), "question": question})
    except Exception as e:
        return f"Erreur IA : {e}"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🚛 LOGI-CASA</h1>", unsafe_allow_html=True)
    st.markdown("---")
    api_key = st.text_input("Clé API OpenAI", type="password", key="openai_api_key")
    st.subheader("🔍 Filtres")
    status_filter = st.multiselect("Statut", ['Pending', 'Delivered', 'Urgent', 'Failed'], default=['Pending', 'Urgent'])
    priority_filter = st.slider("Priorité Min", 1, 3, 1)
    if st.button("🔄 Reset Données"):
        st.session_state.pop('df_points')
        st.rerun()

# --- 6. MAIN ---
filtered_df = df[(df['status'].isin(status_filter)) & (df['priority'] >= priority_filter)]
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🗺️ Supervision Temps Réel")
    
    # CORRECTION ICI : On utilise les constantes globales définies au début
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13, tiles=None)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google', name='Google Maps', overlay=False, control=True
    ).add_to(m)

    if not filtered_df.empty:
        with st.spinner("🧠 Optimisation en cours..."):
            route_geo, total_dist, _ = optimize_route(G, filtered_df)
        
        if route_geo:
            folium.PolyLine(route_geo, color="#4285F4", weight=5, opacity=0.8).add_to(m)
            plugins.AntPath(route_geo, color='black', weight=2, opacity=0.4).add_to(m)

        for _, row in filtered_df.iterrows():
            color = "#28a745" if row['status'] == 'Delivered' else "#dc3545"
            if row['status'] == 'Urgent': color = "#ffc107"
            if row['status'] == 'Failed': color = "#343a40"
            
            folium.CircleMarker(
                [row['lat'], row['lon']], radius=6, color="white", weight=1,
                fill=True, fill_color=color, fill_opacity=1, tooltip=f"{row['name']}"
            ).add_to(m)
        st.success(f"✅ Itinéraire : {total_dist/1000:.2f} km")
    else:
        st.warning("Aucun point ne correspond aux filtres.")

    st_folium(m, width="100%", height=600)

with col2:
    st.markdown("### 📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("À faire", len(filtered_df[filtered_df['status']=='Pending']))
    c2.metric("Urgences", len(filtered_df[filtered_df['status']=='Urgent']))
    
    st.markdown("---")
    user_query = st.text_input("Assistant IA", placeholder="Analyse les urgences...")
    if st.button("Envoyer"):
        if user_query:
            with st.spinner("Analyse..."):
                st.info(ask_agent(user_query, filtered_df))
    
    st.dataframe(filtered_df[['name', 'status']], height=300)