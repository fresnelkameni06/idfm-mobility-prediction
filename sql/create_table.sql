-- ============================================================
-- ÉTAPE 1 — Création de la base de données et de la table
-- Projet : Analyse & Prédiction des flux de mobilité IDFM
-- ============================================================

-- -------------------------------------------------------------
-- POURQUOI ?
-- Avant d'insérer nos 468 226 lignes, MySQL a besoin de savoir
-- comment organiser les données : quelles colonnes, quels types.
-- C'est comme dessiner les cases d'un tableau avant de le remplir.
-- -------------------------------------------------------------

-- 1. Créer la base de données si elle n'existe pas encore
CREATE DATABASE IF NOT EXISTS idfm_mobility
    CHARACTER SET utf8mb4        -- supporte tous les caractères français (accents, etc.)
    COLLATE utf8mb4_unicode_ci;  -- tri et comparaison insensible à la casse

-- 2. Se positionner dans cette base pour la suite
USE idfm_mobility;

-- 3. Supprimer la table si elle existe déjà (utile si on relance le script)
DROP TABLE IF EXISTS validations;

-- 4. Créer la table validations avec toutes ses colonnes
CREATE TABLE validations (

    -- JOUR : la date de la validation (ex: 2025-07-18)
    -- On utilise DATE et pas VARCHAR pour pouvoir faire des calculs de dates
    JOUR DATE NOT NULL,

    -- CODE_STIF_TRNS : code du transporteur (ex: 800 = Transilien)
    -- SMALLINT UNSIGNED = entier positif jusqu'à 65 535, suffisant ici
    CODE_STIF_TRNS SMALLINT UNSIGNED,

    -- CODE_STIF_RES : code du réseau (ex: 803)
    -- VARCHAR car certaines valeurs peuvent contenir des lettres
    CODE_STIF_RES VARCHAR(20),

    -- CODE_STIF_ARRET : code de l'arrêt (ex: 424)
    CODE_STIF_ARRET VARCHAR(20),

    -- ID_ZDC : identifiant de la zone de comptage
    -- BIGINT car les valeurs peuvent être grandes (ex: 59761)
    ID_ZDC BIGINT,

    -- LIBELLE_ARRET : nom de l'arrêt en clair (ex: LARDY, CHATELET)
    -- VARCHAR(100) pour avoir de la marge (max observé : 40 caractères)
    LIBELLE_ARRET VARCHAR(100),

    -- CATEGORIE_TITRE : type de titre utilisé (ex: Forfait Navigo, Imagine R)
    -- VARCHAR(50) pour avoir de la marge (max observé : 28 caractères)
    CATEGORIE_TITRE VARCHAR(50),

    -- NB_VALD : nombre de validations (ex: 1586)
    -- INT UNSIGNED = entier positif jusqu'à ~4 milliards, largement suffisant
    -- (max observé dans les données : 80 638)
    NB_VALD INT UNSIGNED NOT NULL

) ENGINE=InnoDB          -- moteur le plus utilisé, supporte les transactions
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

-- 5. Vérification : afficher la structure de la table créée
DESCRIBE validations

SELECT  SUM(nb_vald)FROM validations where jour ="2026-06-23" and libelle_arret ="CHATELET";

   SELECT MAX(total_jour) AS montant_max
FROM (
    SELECT jour, SUM(nb_vald) AS total_jour
    FROM validations
    WHERE libelle_arret= 'CHATELET'
    GROUP BY jour
) AS t;


SELECT  MAX (jour) FROM validations;

