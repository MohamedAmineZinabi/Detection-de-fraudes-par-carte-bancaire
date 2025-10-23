# Utilise une image Python officielle
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie requirements et installe les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste du projet
COPY . .

# Définit la commande par défaut
CMD ["python", "credit card fraud detection_.py"]
