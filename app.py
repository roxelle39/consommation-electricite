import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
from meteostat import Point, Hourly

# ==============================
# Paramètres
# ==============================
growth_rate = 0.08  # +5% global

# ==============================
# Jours fériés fixes et variables
# ==============================
jours_feries_fixes = {
    "Nouvel An": (1, 1),
    "Fête du Travail": (5, 1),
    "Toussaint": (11, 1),
    "Noël": (12, 25),
    "Assomption": (8, 15)
}

jours_feries_variables = {
    2024: {"Gamou": "2024-09-15", "Ramadan": "2024-04-10", "Tabaski": "2024-06-16", "Korite": "2024-04-04", "Magal": "2024-05-20"},
    2025: {"Gamou": "2025-09-04", "Ramadan": "2025-03-30", "Tabaski": "2025-06-06", "Korite": "2025-03-21", "Magal": "2025-05-09"}
}

# ==============================
# Saison de demande
# ==============================
groupes = {"Bas": [1,2,3,12], "Transition": [4,5,6,11], "Hautes": [7,8,9,10]}
def assigner_saison(mois):
    for g, mlist in groupes.items():
        if mois in mlist:
            return g
    return "Autres"

# ==============================
# Vérifier jours fériés
# ==============================
def check_jours_feries(date):
    is_holiday = is_ramadan = is_tabaski = is_korite = is_gamou = is_magal = 0
    for nom, (m,d) in jours_feries_fixes.items():
        if date.month == m and date.day == d:
            is_holiday = 1
    year = date.year
    if year in jours_feries_variables:
        for nom, date_str in jours_feries_variables[year].items():
            if date == pd.to_datetime(date_str).date():
                is_holiday = 1
                if nom == "Ramadan": is_ramadan=1
                elif nom == "Tabaski": is_tabaski=1
                elif nom == "Korite": is_korite=1
                elif nom == "Gamou": is_gamou=1
                elif nom == "Magal": is_magal=1
    return is_holiday, is_ramadan, is_tabaski, is_korite, is_gamou, is_magal

# ==============================
# Charger modèle
# ==============================
@st.cache_resource
def load_model(jour):
    try:
        return joblib.load(f"model/random_forest_{jour}.pkl")
    except FileNotFoundError:
        st.error(f"Pas de modèle trouvé pour {jour}")
        return None

# ==============================
# Récupérer températures horaires
# ==============================
def fetch_historical_weather(date_select):
    try:
        location = Point(14.7167, -17.4677)  # Dakar
        start = pd.Timestamp(datetime(date_select.year, date_select.month, date_select.day))
        end = start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        df = Hourly(location, start, end).fetch()
        if df.empty or 'temp' not in df.columns:
            return [25 + (h%6) for h in range(24)]
        temps = df['temp'].tolist()
        if len(temps) < 24:
            temps += [temps[-1]]*(24-len(temps))
        elif len(temps) > 24:
            temps = temps[:24]
        temps = [min(t,35) for t in temps]  # limiter à 35°C
        return temps
    except:
        return [25 + (h%6) for h in range(24)]

# ==============================
# Interface Streamlit
# ==============================
st.set_page_config(page_title="Prédiction Consommation Électrique", layout="wide")
st.title("🔌 Prédiction de la Consommation Électrique (24h)")

date_select = st.date_input("📅 Choisir une date :", datetime.today())

if 'temperatures' not in st.session_state:
    st.session_state.temperatures = [0]*24
if st.button("⚡ Récupérer les températures"):
    st.session_state.temperatures = fetch_historical_weather(date_select)

jours = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
jour = jours[date_select.weekday()]
annee = date_select.year
mois = date_select.month
saison = assigner_saison(mois)
profil_type = f"{saison} {jour}"
is_weekend = 1 if jour in ["samedi","dimanche"] else 0
is_holiday, is_ramadan, is_tabaski, is_korite, is_gamou, is_magal = check_jours_feries(date_select)

st.info(f"➡️ Jour : **{jour.capitalize()}**, Saison : **{saison}**, Profil : **{profil_type}**")
st.write(f"📌 Weekend : {is_weekend}, Jour férié : {is_holiday}")
st.write(f"📌 Ramadan: {is_ramadan}, Tabaski: {is_tabaski}, Korité: {is_korite}, Gamou: {is_gamou}, Magal: {is_magal}")

# Affichage températures
heures = list(range(24))
df_temps = pd.DataFrame({"Heure": heures, "Température (°C)": st.session_state.temperatures})
st.subheader("🌡️ Températures horaires")
st.line_chart(df_temps.rename(columns={"Heure":"index"}).set_index("index"))

# ==============================
# Prédiction consommation pour la date saisie + ajustement
# ==============================
if st.button("⚡ Prédire la consommation"):
    model = load_model(jour)
    if model:
        X_input = pd.DataFrame([{
            "year": annee,
            "month": mois,
            "hour": h,
            "temperature": st.session_state.temperatures[h],
            "day_of_week": jours.index(jour),
            "is_weekend": is_weekend,
            "is_ramadan": is_ramadan,
            "is_tabaski": is_tabaski,
            "is_korite": is_korite,
            "is_gamou": is_gamou,
            "is_magal": is_magal,
            "Saison de demande": saison,
            "profil_type": profil_type
        } for h in heures])

        X_input = pd.get_dummies(X_input, columns=["Saison de demande","profil_type"], drop_first=True)
        X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0)

        # Prédictions brutes
        y_pred = model.predict(X_input)

        # Appliquer croissance globale
        y_pred = y_pred * (1 + growth_rate)

        # Réductions ciblées
        for i, h in enumerate(heures):
            if h == 17:
                y_pred[i] *= 0.97  # -4%
            elif h in [8, 9, 10, 11, 12, 13, 14, 18]:
                y_pred[i] *= 0.91  # -8%

        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=heures, y=y_pred, mode='lines+markers', name="Consommation ajustée"))

        fig.update_layout(
            title=f"Prédiction consommation pour le {date_select} ({jour})",
            xaxis=dict(title="Heure"),
            yaxis=dict(title="Consommation (MW)"),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tableau des résultats
        df_result = pd.DataFrame({"Heure": heures, "Température (°C)": st.session_state.temperatures, "Consommation ajustée": y_pred})
        st.subheader("📊 Tableau des résultats")
        st.dataframe(df_result)
