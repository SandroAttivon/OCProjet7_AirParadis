import subprocess
import sys
import webbrowser
import os
import shutil

def run_pytest():
    htmlcov_path = "htmlcov"

    # 🧹 Supprimer le dossier htmlcov/ s'il existe
    if os.path.exists(htmlcov_path) and os.path.isdir(htmlcov_path):
        print("🧹 Suppression du dossier existant htmlcov/...")
        shutil.rmtree(htmlcov_path)

    print("\n🚀 Lancement des tests avec génération de la couverture HTML...\n")

    result = subprocess.run(
        [
            "pytest",
            "--cov=app_with_mlflow",
            "--cov-report=html",
            "--cov-report=term-missing",
            "tests/"
        ],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    if result.returncode != 0:
        print("\n❌ Certains tests ont échoué.\n")
        sys.exit(result.returncode)
    else:
        print("\n✅ Tous les tests sont passés avec succès !")
        print("📁 Rapport HTML généré dans : ./htmlcov/index.html")

        # 🌐 Ouvrir le rapport HTML automatiquement
        report_path = os.path.abspath("htmlcov/index.html")
        webbrowser.open(f"file://{report_path}", new=2)

if __name__ == "__main__":
    run_pytest()