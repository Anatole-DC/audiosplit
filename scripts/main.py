#!/usr/bin/env python3
"""
Lakh MIDI Dataset Explorer - Main Entry Point

This is the main entry point for the Lakh MIDI Dataset exploration project.
Run this file to get started with exploring the dataset.
"""

import sys
import os
from pathlib import Path

def main():
    """Main entry point for the application"""
    print("🎵 Welcome to the Lakh MIDI Dataset Explorer!")
    print("=" * 50)
    print()
    print("This project helps you explore and analyze the Lakh MIDI Dataset,")
    print("a collection of 176,581 unique MIDI files for music research.")
    print()
    print("📋 Available options:")
    print("  [1] Run interactive demo")
    print("  [2] Open Jupyter notebook")
    print("  [3] Quick start guide")
    print("  [4] View project info")
    print("  [5] Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                run_demo()
                break
            elif choice == "2":
                open_notebook()
                break
            elif choice == "3":
                show_quick_start()
                break
            elif choice == "4":
                show_project_info()
                break
            elif choice == "5":
                print("👋 Goodbye! Happy music exploring!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_demo():
    """Run the interactive demo"""
    print("\n🚀 Starting interactive demo...")
    try:
        import demo
        demo.main()
    except ImportError as e:
        print(f"❌ Error importing demo: {e}")
        print("💡 Make sure you have installed all requirements:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def open_notebook():
    """Try to open the Jupyter notebook"""
    print("\n📓 Opening Jupyter notebook...")
    
    notebook_path = Path("lmd_exploration_notebook.ipynb")
    if not notebook_path.exists():
        print("❌ Notebook file not found!")
        return
    
    try:
        import subprocess
        subprocess.run(["jupyter", "notebook", str(notebook_path)], check=True)
    except subprocess.CalledProcessError:
        print("❌ Error opening Jupyter notebook")
        print("💡 Make sure Jupyter is installed:")
        print("   pip install jupyter")
    except FileNotFoundError:
        print("❌ Jupyter not found")
        print("💡 Install Jupyter:")
        print("   pip install jupyter")
        print("💡 Or open the notebook manually in your preferred environment")

def show_quick_start():
    """Show quick start guide"""
    print("\n⚡ Quick Start Guide")
    print("=" * 20)
    print()
    print("1. 📦 Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. 🎵 Basic Usage:")
    print("   from lmd_explorer import LakhMIDIExplorer")
    print("   explorer = LakhMIDIExplorer()")
    print("   explorer.download_dataset(['md5_to_paths', 'match_scores'])")
    print("   explorer.load_metadata()")
    print()
    print("3. 📊 Download and Analyze:")
    print("   explorer.download_dataset(['lmd_matched'])  # ~1.5GB")
    print("   results = explorer.batch_analyze('matched', max_files=100)")
    print("   explorer.generate_statistics()")
    print("   explorer.print_summary_report()")
    print()
    print("4. 📈 Create Visualizations:")
    print("   explorer.create_visualizations()")
    print()
    print("5. 🔬 Advanced Analysis:")
    print("   from analysis_utils import DatasetAnalyzer")
    print("   analyzer = DatasetAnalyzer(explorer)")
    print("   correlations = analyzer.correlation_analysis()")
    print()
    print("💡 Tips:")
    print("   • Start with small max_files values for testing")
    print("   • Use caching to avoid re-processing")
    print("   • Check the README.md for detailed documentation")
    print()
    
    input("Press Enter to continue...")

def show_project_info():
    """Show project information"""
    print("\n📄 Project Information")
    print("=" * 22)
    print()
    print("🎵 Lakh MIDI Dataset Explorer")
    print("   A comprehensive toolkit for exploring Colin Raffel's")
    print("   Lakh MIDI Dataset - one of the largest MIDI collections")
    print("   available for music information retrieval research.")
    print()
    print("📊 Dataset Details:")
    print("   • Total MIDI files: 176,581")
    print("   • Matched files: 45,129") 
    print("   • Dataset size: ~3.5GB (full)")
    print("   • Matched subset: ~1.5GB")
    print()
    print("🔧 Features:")
    print("   • Automatic dataset downloading")
    print("   • Batch MIDI analysis with caching")
    print("   • Statistical analysis and insights")
    print("   • Comprehensive visualizations")
    print("   • Advanced analysis utilities")
    print("   • Export capabilities")
    print()
    print("📚 Files in this project:")
    files_info = [
        ("lmd_explorer.py", "Main explorer class"),
        ("analysis_utils.py", "Advanced analysis tools"),
        ("demo.py", "Interactive demonstration"),
        ("requirements.txt", "Python dependencies"),
        ("lmd_exploration_notebook.ipynb", "Jupyter notebook"),
        ("README.md", "Complete documentation")
    ]
    
    for filename, description in files_info:
        status = "✅" if Path(filename).exists() else "❌"
        print(f"   {status} {filename:<30} - {description}")
    
    print()
    print("🌐 Resources:")
    print("   • Dataset homepage: https://colinraffel.com/projects/lmd/")
    print("   • Pretty MIDI docs: https://craffel.github.io/pretty-midi/")
    print("   • MIR resources: https://musicinformationretrieval.com/")
    print()
    print("📝 Citation:")
    print("   Colin Raffel. 'Learning-Based Methods for Comparing Sequences,")
    print("   with Applications to Audio-to-MIDI Alignment and Matching'.")
    print("   PhD Thesis, 2016.")
    print()
    
    input("Press Enter to continue...")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'pretty_midi', 'tqdm', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print()
        print("💡 Install missing packages:")
        print("   pip install -r requirements.txt")
        print()
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['lmd_data', 'cache']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

if __name__ == "__main__":
    # Check if this is being run directly
    print("🔧 Initializing...")
    
    # Create necessary directories
    create_directories()
    
    # Check requirements (optional - don't block execution)
    requirements_ok = check_requirements()
    if not requirements_ok:
        print("⚠️  Some packages are missing, but you can still explore the project structure.")
        print()
    
    # Run main menu
    main()
