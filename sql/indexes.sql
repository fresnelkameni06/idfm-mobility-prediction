-- ============================================================
-- ÉTAPE 1 — Création des index pour optimiser les requêtes
-- Projet : Analyse & Prédiction des flux de mobilité IDFM
-- ============================================================

-- -------------------------------------------------------------
-- POURQUOI DES INDEX ?
-- Sans index, quand vous faites une requête comme
-- "donne-moi le trafic de CHATELET", MySQL lit les 468 226 lignes
-- une par une pour trouver celles qui correspondent.
-- Avec un index, il va directement au bon endroit — comme l'index
-- d'un livre qui vous dit directement à quelle page chercher.
-- Sur 468K lignes, la différence peut être de plusieurs secondes.
-- -------------------------------------------------------------

USE idfm_mobility;

-- Index sur JOUR
-- Utile pour toutes les requêtes filtrées par date
-- Ex : "donne-moi le trafic du 18 juillet 2025"
CREATE INDEX idx_jour ON validations(JOUR);

-- Index sur LIBELLE_ARRET
-- Utile pour toutes les requêtes filtrées par arrêt
-- Ex : "donne-moi l'historique de CHATELET"
CREATE INDEX idx_arret ON validations(LIBELLE_ARRET);

-- Index sur CATEGORIE_TITRE
-- Utile pour filtrer par type de titre
-- Ex : "donne-moi uniquement les validations Navigo"
CREATE INDEX idx_titre ON validations(CATEGORIE_TITRE);

-- Index combiné (JOUR + LIBELLE_ARRET)
-- C'est l'index le plus important du projet
-- Car la plupart de nos requêtes filtrent à la fois par date ET par arrêt
-- Ex : "trafic de CHATELET le 18 juillet" → utilise cet index directement
CREATE INDEX idx_jour_arret ON validations(JOUR, LIBELLE_ARRET);

-- -------------------------------------------------------------
-- VÉRIFICATION : afficher tous les index créés sur la table
-- -------------------------------------------------------------
SHOW INDEX FROM validations;

-- -------------------------------------------------------------
-- UTILISER EXPLAIN POUR ANALYSER UNE REQUÊTE
-- EXPLAIN montre comment MySQL va exécuter une requête
-- et si elle utilise bien un index ou pas (colonne "key")
-- -------------------------------------------------------------
EXPLAIN SELECT *
FROM validations
WHERE LIBELLE_ARRET = 'CHATELET'
  AND JOUR BETWEEN '2025-07-01' AND '2025-09-30';