import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
from typing import Optional, List, Dict, Any, Union, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import uvicorn
import numpy as np
import requests
import threading
import json
import logging
from contextlib import asynccontextmanager

# Configuration du logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("blood_eligibility_api")

# Modèle global
model = None

# Gestionnaire du cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    global model
    try:
        # Chemin du modèle
        model_path = 'model_eligibilite.pkl'
        
        # Vérifier si le fichier existe
        if not os.path.exists(model_path):
            logger.warning(f"Le fichier modèle {model_path} n'existe pas. Tentative de recherche du modèle...")
            # Recherche d'autres modèles pickle dans le répertoire courant
            pickle_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
            if pickle_files:
                model_path = pickle_files[0]
                logger.info(f"Utilisation du modèle alternatif: {model_path}")
            else:
                logger.error("Aucun fichier modèle pickle trouvé!")
                raise FileNotFoundError("Aucun fichier modèle pickle trouvé!")
        
        # Chargement du modèle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Modèle chargé avec succès depuis {model_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        # En mode développement on peut poursuivre sans erreur fatale
        logger.warning("Lancement de l'API sans modèle chargé (utilisation possible en mode développement)")
    
    yield  # Donne le contrôle à l'application pendant son cycle de vie
    
    # Code exécuté à l'arrêt
    logger.info("Arrêt de l'API - Libération des ressources")

# Initialisation de l'API FastAPI avec cycle de vie
app_fastapi = FastAPI(
    title="API de Prédiction de l'Éligibilité au Don de Sang",
    description="Cette API permet de prédire l'éligibilité d'une personne au don de sang en fonction de diverses caractéristiques démographiques et médicales.",
    version="2.0",
    docs_url="/docs",
    lifespan=lifespan
)

# Création du modèle de données avancé pour prendre en compte toutes les variables
class Donneur_Data(BaseModel):
    # Données démographiques
    Genre: str = Field(
        description="Genre du donneur", 
        examples=["Homme", "Femme"],
        default="Homme"
    )
    
    Age: int = Field(
        description="Âge du donneur", 
        ge=18, le=65,
        default=30
    )
    
    Niveau_etude: str = Field(
        description="Niveau d'étude du donneur",
        examples=["Primaire", "Secondaire", "Universitaire", "Aucun", "Pas Précisé"],
        default="Universitaire"
    )
    
    Situation_Matrimoniale: str = Field(
        description="Situation matrimoniale du donneur",
        examples=["Célibataire", "Marié (e)", "Divorcé(e)", "veuf (veuve)"],
        default="Célibataire"
    )
    
    Religion: str = Field(
        description="Religion du donneur",
        examples=["chrétien (catholique)", "chrétien (protestant)", "Musulman"],
        default="chrétien (catholique)"
    )
    
    Profession: str = Field(
        description="Profession du donneur",
        examples=["Étudiant-Eleve", "Enseignant", "Commerçant", "Informaticien", "Sans emploi"],
        default="Étudiant-Eleve"
    )
    
    Taille: float = Field(
        description="Taille du donneur en cm",
        ge=140, le=210,
        default=170.0
    )
    
    Poids: float = Field(
        description="Poids du donneur en kg",
        ge=45, le=150,
        default=70.0
    )
    
    Arrondissement_de_residence: str = Field(
        description="Arrondissement de résidence du donneur",
        examples=["douala 1", "douala 2", "douala 3", "douala 4", "douala 5"],
        default="douala 1"
    )
    
    Quartier_residence: str = Field(
        description="Quartier de résidence du donneur",
        examples=["Bonapriso", "Makepe", "Ndogbong", "Akwa"],
        default="Bonapriso"
    )
    
    Nationalite: str = Field(
        description="Nationalité du donneur",
        examples=["Camerounaise", "Etranger"],
        default="Camerounaise"
    )
    
    # Données médicales générales
    Taux_hemoglobine: float = Field(
        description="Taux d'hémoglobine du donneur (g/dL)",
        ge=8.0, le=20.0,
        default=14.0
    )
    
    A_deja_donne: str = Field(
        description="Le donneur a-t-il déjà donné du sang?",
        examples=["Oui", "Non"],
        default="Non"
    )
    
    Date_dernier_don: Optional[str] = Field(
        description="Date du dernier don au format MM/YYYY (Mois/Année)",
        examples=["01/2024", "12/2023"],
        default=None
    )
    
    # Nouvelles variables médicales
    Anti_biotherapie: str = Field(
        description="Le donneur est-il sous anti-biothérapie?",
        examples=["oui", "non"],
        default="non"
    )
    
    Taux_hemoglobine_bas: Optional[str] = Field(
        description="Le taux d'hémoglobine est-il bas?",
        examples=["oui", "non"],
        default=None
    )
    
    IST_recente: str = Field(
        description="Le donneur a-t-il eu une IST récente (excluant VIH, Hbs, Hcv)?",
        examples=["oui", "non"],
        default="non"
    )
    
    # Variables spécifiques pour les femmes
    
    DDR_recent: Optional[str] = Field(
        description="La dernière date des règles est-elle inférieure à 14 jours avant le don?",
        examples=["oui", "non"],
        default=None
    )
    
    Allaitement: Optional[str] = Field(
        description="La donneuse allaite-t-elle?",
        examples=["oui", "non"],
        default=None
    )
    
    Accouchement_recent: Optional[str] = Field(
        description="La donneuse a-t-elle accouché dans les 6 derniers mois?",
        examples=["oui", "non"],
        default=None
    )
    
    Interruption_grossesse_recente: Optional[str] = Field(
        description="La donneuse a-t-elle eu une interruption de grossesse dans les 6 derniers mois?",
        examples=["oui", "non"],
        default=None
    )
    
    Enceinte: Optional[str] = Field(
        description="La donneuse est-elle enceinte?",
        examples=["oui", "non"],
        default=None
    )
    
    Autre_raison_indispo_femme: Optional[str] = Field(
        description="Autre raison d'indisponibilité spécifique aux femmes",
        default=None
    )
    
    # Conditions médicales générales
    Antecedent_transfusion: str = Field(
        description="Le donneur a-t-il des antécédents de transfusion?",
        examples=["oui", "non"],
        default="non"
    )
    
    Porteur_HIV_Hbs_Hcv: str = Field(
        description="Le donneur est-il porteur de HIV, Hbs ou Hcv?",
        examples=["oui", "non"],
        default="non"
    )
    
    Opere: str = Field(
        description="Le donneur a-t-il été opéré récemment?",
        examples=["oui", "non"],
        default="non"
    )
    
    Drepanocytaire: str = Field(
        description="Le donneur est-il drépanocytaire?",
        examples=["oui", "non"],
        default="non"
    )
    
    Diabetique: str = Field(
        description="Le donneur est-il diabétique?",
        examples=["oui", "non"],
        default="non"
    )
    
    Hypertendu: str = Field(
        description="Le donneur est-il hypertendu?",
        examples=["oui", "non"],
        default="non"
    )
    
    Asthmatique: str = Field(
        description="Le donneur est-il asthmatique?",
        examples=["oui", "non"],
        default="non"
    )
    
    Cardiaque: str = Field(
        description="Le donneur a-t-il des problèmes cardiaques?",
        examples=["oui", "non"],
        default="non"
    )
    
    Tatoue: str = Field(
        description="Le donneur est-il tatoué?",
        examples=["oui", "non"],
        default="non"
    )
    
    Scarifie: str = Field(
        description="Le donneur est-il scarifié?",
        examples=["oui", "non"],
        default="non"
    )
    
    Autre_raison_ineligible: Optional[str] = Field(
        description="Autre raison d'inéligibilité",
        default=None
    )
    
    
    
    # Validation des dates et calculs dépendants
    @validator('Taux_hemoglobine_bas', pre=True, always=True)
    def determine_hemoglobine_bas(cls, v, values):
        if v is not None:
            return v
        
        if 'Taux_hemoglobine' in values and values['Taux_hemoglobine'] is not None:
            return "Oui" if values['Taux_hemoglobine'] < 12.0 else "non"
        
        return "non"  # Valeur par défaut

    model_config = {
        "json_schema_extra": {
            "example": {
                "Genre": "Homme",
                "Age": 30,
                "Niveau_etude": "Universitaire",
                "Situation_Matrimoniale": "Célibataire",
                "Religion": "chrétien (catholique)",
                "Profession": "Informaticien",
                "Taille": 175.0,
                "Poids": 70.0,
                "Arrondissement_de_residence": "douala 1",
                "Quartier_residence": "Bonapriso",
                "Nationalite": "Camerounaise",
                "Taux_hemoglobine": 14.0,
                "A_deja_donne": "non",
                "Date_dernier_don": "",
                "Anti_biotherapie": "non",
                "IST_recente": "non",
                "Antecedent_transfusion": "non",
                "Porteur_HIV_Hbs_Hcv": "non",
                "Opere": "non",
                "Drepanocytaire": "non",
                "Diabetique": "non",
                "Hypertendu": "non",
                "Asthmatique": "non", 
                "Cardiaque": "non",
                "Tatoue": "non",
                "Scarifie": "non"
            }
        }
    }


# Classe de résultat améliorée pour la prédiction multiclasse
class PredictionModel(BaseModel):
    statut: Literal["Eligible", "Temporairement Non-eligible", "Définitivement non-eligible"] = Field(
        ..., 
        description="Statut d'éligibilité au don de sang"
    )
    probabilites: Dict[str, float] = Field(
        ..., 
        description="Probabilités pour chaque classe (entre 0 et 1)"
    )
    classe_majoritaire: str = Field(
        ..., 
        description="Classe avec la probabilité la plus élevée"
    )
    message: str = Field(
        ..., 
        description="Message explicatif"
    )
    raisons: Optional[List[str]] = Field(
        None, 
        description="Raisons expliquant le statut d'éligibilité"
    )
    recommandations: Optional[List[str]] = Field(
        None, 
        description="Recommandations personnalisées basées sur le profil"
    )


# Fonction de préparation des données pour la prédiction
def prepare_data_for_prediction(data: Donneur_Data):
    """
    Prépare les données pour la prédiction en les transformant en DataFrame
    avec le format attendu par le modèle
    
    Parameters:
    -----------
    data : Donneur_Data
        Les données du donneur à préparer
        
    Returns:
    --------
    tuple
        (DataFrame préparé pour le modèle, DataFrame avec les valeurs originales, DataFrame avec les variables dérivées)
    """
    try:
        # Conversion en dictionnaire puis en DataFrame
        data_dict = data.dict()
        df_original = pd.DataFrame([data_dict])
        
        # Mappage des noms de colonnes pour correspondre au format attendu par le modèle
        mapping = {
            "Genre": "Genre",
            "Age": "Age",
            "Niveau_etude": "Niveau d'etude",
            "Situation_Matrimoniale": "Situation Matrimoniale (SM)",
            "Religion": "Religion",
            "A_deja_donne": "A-t-il (elle) déjà donné le sang",
            "Profession": "Profession_Commune",
            "Taille": "Taille",
            "Poids": "Poids",
            "Quartier_residence": "Quartier de Résidence",
            "Arrondissement_de_residence": "Arr",
            "Nationalite": "Nationalité",
            "Taux_hemoglobine": "Taux d'hémoglobine",
            "Anti_biotherapie": "Est sous anti-biothérapie",
            "Taux_hemoglobine_bas": "Taux d'hémoglobine bas",
            "IST_recente": "IST récente (Exclu VIH, Hbs, Hcv)",
            "DDR_recent": "La DDR est mauvais si <14 jour avant le don",
            "Allaitement": "Allaitement",
            "Accouchement_recent": "A accoucher ces 6 derniers mois",
            "Interruption_grossesse_recente": "Interruption de grossesse  ces 06 derniers mois",
            "Enceinte": "est enceinte",
            "Antecedent_transfusion": "Antécédent de transfusion",
            "Porteur_HIV_Hbs_Hcv": "Porteur(HIV,hbs,hcv)",
            "Opere": "Opéré",
            "Drepanocytaire": "Drepanocytaire",
            "Diabetique": "Diabétique",
            "Hypertendu": "Hypertendus",
            "Asthmatique": "Asthmatiques",
            "Cardiaque": "Cardiaque",
            "Tatoue": "Tatoué",
            "Scarifie": "Scarifié",
            "Autre_raison_indispo_femme": "Autre_raison_indispo_femme",
            "Autre_raison_ineligible": "Autre_raison_ineligible"
        }

        # Renommer les colonnes
        df = df_original.rename(columns=mapping)
        
        # Créer et initialiser la colonne du dernier don < 3 mois
        df["date de dernier Don < 3 mois"] = "non"
        
        # Vérifier si le dernier don est récent (moins de 3 mois)
        if data.Date_dernier_don and data.A_deja_donne == "Oui":
            try:
                # Format attendu: MM/YYYY
                month, year = map(int, data.Date_dernier_don.split('/'))
                last_donation_date = datetime(year, month, 1)
                
                # Date actuelle
                current_date = datetime.now()
                
                # Vérifier si le dernier don est inférieur à 3 mois
                if (current_date - last_donation_date).days < 90:  # environ 3 mois
                    df["date de dernier Don < 3 mois"] = "oui"
            except Exception as e:
                logger.warning(f"Erreur lors du calcul de la date du dernier don: {e}")
                # Ne pas échouer complètement en cas d'erreur dans le calcul de date
        
        def convert_to_boolean(value):
            if pd.isna(value) or value == 'non' or value == 'Non':
                return 0
            else:
                return 1
                
        # Liste des colonnes booléennes à convertir
        bool_columns = [
            'Est sous anti-biothérapie', 'Taux d’hémoglobine bas',
            'IST récente (Exclu VIH, Hbs, Hcv)', 'La DDR est mauvais si <14 jour avant le don',
            'Allaitement', 'A accoucher ces 6 derniers mois', 'Interruption de grossesse  ces 06 derniers mois',
            'est enceinte', 'Antécédent de transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
            'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques', 'Cardiaque',
            'Tatoué', 'Scarifié', 'Autre_raison_indispo_femme', 'Autre_raison_ineligible',
            'date de dernier Don < 3 mois'  # Assurez-vous que cette colonne est incluse
        ]
        
        # Convertir les colonnes en booléens (0 ou 1)
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].apply(convert_to_boolean)
        
        # Listes des colonnes pour les groupes
        indispo_commune_columns = [
            'date de dernier Don < 3 mois',
            'Est sous anti-biothérapie', 'Taux d’hémoglobine bas',
            'IST récente (Exclu VIH, Hbs, Hcv)'
        ]

        indispo_femme_columns = [
            'La DDR est mauvais si <14 jour avant le don', 'Allaitement',
            'A accoucher ces 6 derniers mois', 'Interruption de grossesse  ces 06 derniers mois',
            'est enceinte', 'Autre_raison_indispo_femme'
        ]

        ineligible_columns = [
            'Antécédent de transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré', 'Drepanocytaire',
            'Diabétique', 'Hypertendus', 'Asthmatiques', 'Cardiaque', 'Tatoué', 'Scarifié',
            'Autre_raison_ineligible'
        ]

        # Calculer les sommes pour chaque groupe de variables
        # Vérifier d'abord que toutes les colonnes existent
        existing_indispo_commune = [col for col in indispo_commune_columns if col in df.columns]
        existing_indispo_femme = [col for col in indispo_femme_columns if col in df.columns]
        existing_ineligible = [col for col in ineligible_columns if col in df.columns]
        
        # Calculer les sommes seulement pour les colonnes qui existent
        if existing_indispo_commune:
            df['somme_indispo_commune'] = df[existing_indispo_commune].sum(axis=1)
        else:
            df['somme_indispo_commune'] = 0
            
        if existing_indispo_femme:
            df['somme_indispo_femme'] = df[existing_indispo_femme].sum(axis=1)
        else:
            df['somme_indispo_femme'] = 0
            
        if existing_ineligible:
            df['somme_ineligible'] = df[existing_ineligible].sum(axis=1)
        else:
            df['somme_ineligible'] = 0
        
        # Variables spécifiques aux femmes
        female_specific_cols = indispo_femme_columns
        
        # Colonnes numériques
        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col not in bool_columns + ['somme_indispo_commune', 'somme_indispo_femme', 'somme_ineligible', 'ÉLIGIBILITÉ AU DON.']:
                numeric_cols.append(col)
        
        # Gestion spécifique pour le genre
        gender_column = 'Genre'
        
        if gender_column in df.columns:
            # Mettre à 0 les colonnes spécifiques aux femmes pour les hommes
            is_female = df[gender_column].isin(['F', 'f', 'Femme', 'femme'])
            
            for col in female_specific_cols:
                if col in df.columns:
                    df.loc[~is_female, col] = 0
        
        # Colonnes catégorielles
        categorical_cols = []
        for col in df.columns:
            if col not in numeric_cols and col not in bool_columns + ['somme_indispo_commune', 'somme_indispo_femme', 'somme_ineligible']:
                categorical_cols.append(col)
        
        # S'assurer que toutes les colonnes nécessaires sont présentes
        required_columns = [
            "Niveau d'etude", 'Genre', 'Taille', 'Poids',
            'Situation Matrimoniale (SM)', 'Profession_Commune',
            'Arr', 'Quartier de Résidence',
            'Nationalité', 'Religion', 'A-t-il (elle) déjà donné le sang',
            'Taux d’hémoglobine', 'Age', 'Est sous anti-biothérapie',
            'Taux d’hémoglobine bas', 'date de dernier Don < 3 mois',
            'IST récente (Exclu VIH, Hbs, Hcv)', 'La DDR est mauvais si <14 jour avant le don',
            'Allaitement', 'A accoucher ces 6 derniers mois',
            'Interruption de grossesse  ces 06 derniers mois', 'est enceinte',
            'Antécédent de transfusion', 'Porteur(HIV,hbs,hcv)',
            'Opéré', 'Drepanocytaire', 'Diabétique', 'Hypertendus',
            'Asthmatiques', 'Cardiaque', 'Tatoué', 'Scarifié',
            'Autre_raison_indispo_femme', 'Autre_raison_ineligible',
            'somme_indispo_commune', 'somme_indispo_femme', 'somme_ineligible'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                if col in bool_columns:
                    df[col] = 0  # Valeur par défaut pour les colonnes booléennes manquantes
                else:
                    df[col] = None  # Valeur par défaut pour les autres colonnes manquantes
        
        logger.info(f"DataFrame préparé avec succès: {df.shape}")
        
        return df, df_original
    
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données: {e}")
        raise
def generate_raisons_et_recommandations(data: Donneur_Data, statut_predit: str) -> tuple:
    """
    Génère les raisons explicatives et les recommandations personnalisées en fonction du profil et du statut prédit.
    
    Parameters:
    -----------
    data : Donneur_Data
        Les données du donneur
    statut_predit : str
        Le statut d'éligibilité prédit par le modèle
        
    Returns:
    --------
    tuple
        (raisons, recommandations) contenant les listes de raisons et recommandations
    """
    raisons = []
    recommandations = []
    
    # Raisons communes de non-éligibilité/indisponibilité
    if statut_predit == "Définitivement non-eligible":
        # Causes définitives d'inéligibilité
        if data.Antecedent_transfusion == "oui":
            raisons.append("Antécédent de transfusion sanguine")
            recommandations.append("Les personnes ayant reçu une transfusion sanguine ne peuvent généralement pas donner leur sang à cause des risques potentiels.")
        
        if data.Porteur_HIV_Hbs_Hcv == "oui":
            raisons.append("Porteur de HIV, Hbs ou Hcv")
            recommandations.append("Les porteurs de certains virus transmissibles par le sang ne peuvent pas faire de don pour des raisons de sécurité.")
        
        if data.Drepanocytaire == "oui":
            raisons.append("Présence de drépanocytose")
            recommandations.append("La drépanocytose est une contre-indication permanente au don de sang.")
        
        if data.Diabetique == "oui":
            raisons.append("Diabète")
            recommandations.append("Le diabète peut être une contre-indication au don de sang selon sa gravité et son traitement.")
            
        if data.Age < 18 or data.Age > 65:
            raisons.append(f"Âge incompatible avec le don de sang ({data.Age} ans)")
            recommandations.append("L'âge légal pour donner son sang se situe entre 18 et 65 ans.")
            
    elif statut_predit == "Temporairement Non-eligible":
        # Causes temporaires d'indisponibilité
        if data.Anti_biotherapie == "oui":
            raisons.append("Traitement antibiotique en cours")
            recommandations.append("Vous pourrez donner votre sang après la fin de votre traitement antibiotique et un délai supplémentaire (généralement 7 à 14 jours).")
            
        if data.Taux_hemoglobine < 12.0:
            raisons.append(f"Taux d'hémoglobine trop bas ({data.Taux_hemoglobine} g/dL)")
            recommandations.append("Consommez des aliments riches en fer (viande rouge, lentilles, épinards) et consultez votre médecin si ce taux persiste.")
            
        if data.IST_recente == "oui":
            raisons.append("IST récente")
            recommandations.append("Après guérison d'une IST, un délai d'attente est nécessaire avant de pouvoir donner son sang.")
            
        if data.Opere == "oui":
            raisons.append("Opération récente")
            recommandations.append("Après une opération, un délai d'attente (généralement de 4 à 6 mois) est nécessaire avant de pouvoir donner son sang.")
            
        if data.Genre == "Femme":
            if data.Enceinte == "oui":
                raisons.append("Grossesse en cours")
                recommandations.append("Le don de sang n'est pas recommandé pendant la grossesse. Vous pourrez donner votre sang 6 mois après l'accouchement.")
                
            if data.Allaitement == "oui":
                raisons.append("Allaitement en cours")
                recommandations.append("L'allaitement peut être une contre-indication temporaire au don de sang.")
                
            if data.Accouchement_recent == "oui":
                raisons.append("Accouchement récent (< 6 mois)")
                recommandations.append("Vous pourrez donner votre sang 6 mois après l'accouchement.")
                
            if data.Interruption_grossesse_recente == "oui":
                raisons.append("Interruption de grossesse récente (< 6 mois)")
                recommandations.append("Vous pourrez donner votre sang 6 mois après l'interruption de grossesse.")
                
            if data.DDR_recent == "oui":
                raisons.append("Menstruation récente (< 14 jours)")
                recommandations.append("Il est préférable d'attendre quelques jours après la fin de vos règles pour donner votre sang.")
                
        # Vérification poids
        if data.Poids < 50:
            raisons.append(f"Poids insuffisant ({data.Poids} kg)")
            recommandations.append("Le poids minimum recommandé pour le don de sang est de 50 kg.")
            
    else:  # Eligible
        # Pas de raison négative, mais des recommandations pour le don
        recommandations.append("Hydratez-vous bien (au moins 500 ml d'eau) avant votre don.")
        recommandations.append("Mangez normalement avant le don, évitez de venir à jeun.")
        recommandations.append("Apportez une pièce d'identité et, si possible, votre carte de groupe sanguin.")
        recommandations.append("Prévoyez environ une heure sur place pour l'ensemble du processus de don.")
        
        if data.A_deja_donne == "Non":
            recommandations.append("Pour un premier don, vous aurez un entretien médical confidentiel pour vérifier votre éligibilité.")
            
    return raisons, recommandations


@app_fastapi.post("/predict", response_model=PredictionModel)
async def predict(data: Donneur_Data):
    """
    Prédit l'éligibilité au don de sang en fonction des données fournies
    
    Parameters:
    -----------
    data : Donneur_Data
        Les données du donneur
        
    Returns:
    --------
    PredictionModel
        Le résultat de la prédiction multiclasse
    """
    try:
        # Préparation des données
        df, df_original = prepare_data_for_prediction(data)
        
        logger.info("Données préparées, lancement de la prédiction multiclasse...")
        
        # Prédiction avec le modèle
        try:
            if model is None:
                raise ValueError("Le modèle n'est pas chargé")
                
            # Prédiction des probabilités pour chaque classe
            probas = model.predict_proba(df)
            
            # Les classes devraient être ordonnées comme:
            # 'Eligible', 'Temporairement Non-eligible', 'Définitivement non-eligible'
            # Mais vérifions pour être sûr
            classes = model.classes_
            
            # Créer un dictionnaire des probabilités avec le nom des classes
            proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probas[0])}
            
            # Déterminer la classe prédite (celle avec la probabilité la plus élevée)
            predicted_class_idx = np.argmax(probas[0])
            predicted_class = str(classes[predicted_class_idx])
            
            logger.info(f"Classe prédite: {predicted_class} avec probabilité {proba_dict[predicted_class]:.4f}")
            
        except Exception as model_error:
            logger.error(f"Erreur lors de la prédiction avec le modèle: {model_error}")
            
            # Prédiction par règles en cas d'erreur du modèle
            logger.info("Utilisation des règles métier pour la prédiction...")
            
            # Variables pour compter les facteurs d'inéligibilité/indisponibilité
            facteurs_definitive = 0
            facteurs_temporaire = 0
            
            # Facteurs d'inéligibilité définitive
            if (data.Antecedent_transfusion == "oui" or 
                data.Porteur_HIV_Hbs_Hcv == "oui" or 
                data.Drepanocytaire == "oui" or 
                data.Diabetique == "oui" or
                data.Age < 18 or 
                data.Age > 65):
                facteurs_definitive += 1
            
            # Facteurs d'indisponibilité temporaire
            if (data.Anti_biotherapie == "oui" or 
                data.Taux_hemoglobine < 12.0 or 
                data.IST_recente == "oui" or 
                data.Opere == "oui" or
                data.Poids < 50):
                facteurs_temporaire += 1
                
            # Facteurs spécifiques aux femmes
            if data.Genre == "Femme":
                if (data.Enceinte == "Oui" or 
                    data.Allaitement == "Oui" or 
                    data.Accouchement_recent == "oui" or 
                    data.Interruption_grossesse_recente == "oui" or
                    data.DDR_recent == "oui"):
                    facteurs_temporaire += 1
            
            # Détermination de la classe
            if facteurs_definitive > 0:
                predicted_class = "Définitivement non-eligible"
                proba_dict = {
                    "Eligible": 0.05,
                    "Temporairement Non-eligible": 0.15,
                    "Définitivement non-eligible": 0.8
                }
            elif facteurs_temporaire > 0:
                predicted_class = "Temporairement Non-eligible"
                proba_dict = {
                    "Eligible": 0.1,
                    "Temporairement Non-eligible": 0.8,
                    "Définitivement non-eligible": 0.1
                }
            else:
                predicted_class = "Eligible"
                proba_dict = {
                    "Eligible": 0.9,
                    "Temporairement Non-eligible": 0.05,
                    "Définitivement non-eligible": 0.05
                }
                
            logger.info(f"Classe prédite par règles: {predicted_class}")
        
        # Génération des raisons et recommandations personnalisées
        raisons, recommandations = generate_raisons_et_recommandations(data, predicted_class)
        
        # Génération du message en fonction du statut prédit
        if predicted_class == "Eligible":
            message = "Vous êtes éligible au don de sang. Merci pour votre générosité!"
            if proba_dict[predicted_class] > 0.9:
                message += " Votre profil est idéal pour le don de sang."
            elif proba_dict[predicted_class] > 0.8:
                message += " Votre profil est très favorable pour le don de sang."
            else:
                message += " Votre profil est compatible avec le don de sang."
                
        elif predicted_class == "Temporairement Non-eligible":
            message = "Vous êtes temporairement non éligible au don de sang."
            if raisons:
                message += " Les raisons principales sont liées à des facteurs temporaires qui peuvent évoluer."
            else:
                message += " Notre modèle d'analyse a détecté des facteurs temporaires qui pourraient affecter la qualité du don."
                
        else:  # Définitivement non-eligible
            message = "Vous n'êtes pas éligible au don de sang de façon permanente."
            if raisons:
                message += " Cette décision est basée sur des critères médicaux stricts pour la sécurité des donneurs et des receveurs."
            else:
                message += " Notre modèle d'analyse a détecté des facteurs permanents qui empêchent le don de sang."
        
        # Préparation de la réponse
        response = PredictionModel(
            statut=predicted_class,
            probabilites=proba_dict,
            classe_majoritaire=predicted_class,
            message=message,
            raisons=raisons if raisons else None,
            recommandations=recommandations if recommandations else None
        )
        
        logger.info(f"Prédiction réussie: {predicted_class} (Probabilité: {proba_dict[predicted_class]:.2f})")
        
        return response
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")


@app_fastapi.get("/")
def root():
    """
    Route racine de l'API
    
    Returns:
    --------
    dict
        Message de bienvenue
    """
    return {
        "message": "Bienvenue sur l'API pour la prédiction du statut d'éligibilité au don de sang",
        "documentation": "/docs",
        "version": "2.0",
        "status": "active"
    }


@app_fastapi.get("/health")
def health_check():
    """
    Vérifie l'état de santé de l'API
    
    Returns:
    --------
    dict
        État de santé de l'API
    """
    try:
        # Vérification que le modèle est chargé
        if 'model' not in globals() or model is None:
            return {
                "status": "warning",
                "message": "L'API est en ligne mais le modèle n'est pas chargé",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "status": "healthy",
            "message": "L'API est en ligne et le modèle est chargé correctement",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur de santé: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


