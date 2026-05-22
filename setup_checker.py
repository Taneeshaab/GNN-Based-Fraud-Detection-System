"""
Setup Verification Script for Fraud Detection System
Run this before the main demo to ensure everything is configured correctly
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        print(f"   Install with: pip install {package_name}")
        return False

def check_neo4j_connection(uri, user, password):
    """Check Neo4j connection"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        driver.close()
        print(f"✅ Neo4j connection successful!")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print(f"   Check that Neo4j is running and credentials are correct")
        return False

def main():
    print("="*60)
    print("FRAUD DETECTION SYSTEM - SETUP VERIFICATION")
    print("="*60)
    print()
    
    print("Checking Python version...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"⚠️  Python {python_version.major}.{python_version.minor} detected")
        print(f"   Recommended: Python 3.8 or higher")
    print()
    
    print("Checking required packages...")
    packages = [
        ('neo4j', 'neo4j'),
        ('torch', 'torch'),
        ('torch-geometric', 'torch_geometric'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scikit-learn', 'sklearn'),
        ('networkx', 'networkx'),
    ]
    
    all_installed = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_installed = False
    print()
    
    if not all_installed:
        print("⚠️  Some packages are missing. Install them with:")
        print("   pip install neo4j torch torch-geometric pandas numpy matplotlib seaborn scikit-learn networkx")
        print()
    
    print("Checking Neo4j connection...")
    print("Enter your Neo4j credentials:")
    uri = input("  URI (default: bolt://localhost:7687): ").strip() or "bolt://localhost:7687"
    user = input("  Username (default: neo4j): ").strip() or "neo4j"
    password = input("  Password: ").strip()
    
    if password:
        neo4j_ok = check_neo4j_connection(uri, user, password)
        print()
        
        if neo4j_ok:
            print("="*60)
            print("✅ ALL CHECKS PASSED!")
            print("="*60)
            print()
            print("Your configuration:")
            print(f"  URI: {uri}")
            print(f"  User: {user}")
            print(f"  Password: {'*' * len(password)}")
            print()
            print("Update fraud_detection_system.py with these credentials:")
            print(f'  neo4j = Neo4jConnector(')
            print(f'      uri="{uri}",')
            print(f'      user="{user}",')
            print(f'      password="{password}"')
            print(f'  )')
            print()
            print("You're ready to run the demo! 🚀")
        else:
            print("="*60)
            print("⚠️  NEO4J CONNECTION FAILED")
            print("="*60)
            print()
            print("Troubleshooting steps:")
            print("1. Make sure Neo4j is running")
            print("   - Open Neo4j Desktop and start your database")
            print("   - Or run: neo4j start")
            print("2. Verify credentials in Neo4j Browser (localhost:7474)")
            print("3. Check firewall settings")
            print("4. Try restarting Neo4j")
    else:
        print("⚠️  Password not provided. Skipping Neo4j connection test.")
    
    print()

if __name__ == "__main__":
    main()