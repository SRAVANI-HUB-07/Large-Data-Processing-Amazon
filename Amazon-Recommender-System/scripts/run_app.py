#!/usr/bin/env python3
"""
Enhanced Run script for Amazon Recommender System with Collaborative Filtering
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import AmazonRecommenderApp
from src.utils.helpers import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Amazon Recommender System')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode with sample queries')
    
    args = parser.parse_args()
    
    setup_logging()
    
    app = AmazonRecommenderApp()
    
    try:
        app.initialize_data()
        
        if args.interactive:
            run_interactive_mode(app)
        elif args.demo:
            run_demo_mode(app)
        else:
            print("Application running in background mode")
            print("Use --interactive for interactive mode or --demo for demo")
            import time
            while True:
                time.sleep(10)
                
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)
    finally:
        app.shutdown()

def run_interactive_mode(app):
    """Run enhanced interactive mode with collaborative filtering"""
    print("\n" + "="*60)
    print("Amazon Recommender System - Enhanced Interactive Mode")
    print("="*60)
    print_help()
    
    while True:
        try:
            command = input("\nEnter command (type 'help' for options): ").strip().split()
            
            if not command:
                continue
                
            cmd_type = command[0].lower()
            
            if cmd_type == 'quit' or cmd_type == 'exit':
                break
                
            elif cmd_type == 'help':
                print_help()
                
            elif cmd_type == 'search':
                handle_search_command(app, command[1:])
                
            elif cmd_type == 'recommend':
                if len(command) > 1:
                    user_id = command[1]
                    n = int(command[2]) if len(command) > 2 else 10
                    print(f"Getting hybrid recommendations for user {user_id}...")
                    recs = app.get_recommendations(user_id=user_id, n=n)
                    if recs.count() > 0:
                        print(f"Recommended {recs.count()} books:")
                        recs.select("product_id", "title", "category", "price", "average_rating").show(n, truncate=50)
                    else:
                        print("No recommendations found")
                else:
                    print("Usage: recommend [user_id] [n=10]")
                    print("Example: recommend CUST_00001 10")
                    
            elif cmd_type == 'collab':
                if len(command) > 1:
                    user_id = command[1]
                    n = int(command[2]) if len(command) > 2 else 10
                    print(f"Getting collaborative recommendations for user {user_id}...")
                    recs = app.get_collaborative_recommendations(user_id=user_id, n=n)
                    if recs.count() > 0:
                        print(f"Collaborative recommendations ({recs.count()} books):")
                        recs.select("product_id", "title", "category", "recommendation_score", "average_rating").show(n, truncate=50)
                    else:
                        print("No collaborative recommendations found")
                else:
                    print("Usage: collab [user_id] [n=10]")
                    print("Example: collab CUST_00001 10")
                    
            elif cmd_type == 'also_bought':
                if len(command) > 1:
                    product_id = command[1]
                    n = int(command[2]) if len(command) > 2 else 10
                    print(f"Finding products also bought with {product_id}...")
                    recs = app.get_also_bought(product_id, n)
                    if recs.count() > 0:
                        print(f"Customers who bought this also bought ({recs.count()} books):")
                        recs.select("product_id", "title", "category", "similarity_score", "average_rating").show(n, truncate=50)
                    else:
                        print("No 'also bought' recommendations found")
                else:
                    print("Usage: also_bought [product_id] [n=10]")
                    print("Example: also_bought 014241543X 10")
                    
            elif cmd_type == 'similar':
                if len(command) > 1:
                    product_id = command[1]
                    n = int(command[2]) if len(command) > 2 else 10
                    print(f"Finding books similar to {product_id}...")
                    similar = app.get_recommendations(product_id=product_id, n=n)
                    if similar.count() > 0:
                        print(f"Found {similar.count()} similar books:")
                        similar.select("product_id", "title", "category", "price", "average_rating").show(n, truncate=50)
                    else:
                        print("No similar books found")
                else:
                    print("Usage: similar [product_id] [n=10]")
                    print("Example: similar 014241543X 10")
                    
            elif cmd_type == 'stats':
                print("Loading category statistics...")
                stats = app.get_category_stats()
                stats.show(truncate=False)
                
            elif cmd_type == 'product':
                if len(command) > 1:
                    product_id = command[1]
                    print(f"Getting details for product {product_id}...")
                    product = app.get_product_details(product_id)
                    if product.count() > 0:
                        product.select("product_id", "title", "category", "price", "description", 
                                     "average_rating", "review_count", "sales_rank").show(truncate=100)
                    else:
                        print("Product not found")
                else:
                    print("Usage: product [product_id]")
                    print("Example: product 014241543X")
                    
            elif cmd_type == 'user':
                if len(command) > 1:
                    user_id = command[1]
                    print(f"Getting history for user {user_id}...")
                    history = app.get_user_history(user_id)
                    if history.count() > 0:
                        print(f"Review history ({history.count()} reviews):")
                        history.select("product_id", "title", "category", "rating", "review_date").show(truncate=50)
                    else:
                        print("No review history found")
                else:
                    print("Usage: user [user_id]")
                    print("Example: user CUST_00001")
                    
            elif cmd_type == 'copurchasers':
                if len(command) > 2:
                    user_id = command[1]
                    product_id = command[2]
                    print(f"Finding number of customers purchasing same product as user {user_id}...")
                    count = app.get_copurchasers_count(user_id, product_id)
                    print(f"Number of customers purchasing product {product_id} (excluding user {user_id}): {count}")
                else:
                    print("Usage: copurchasers [user_id] [product_id]")
                    print("Example: copurchasers CUST_00001 014241543X")
                    
            elif cmd_type == 'eval_collab':
                print("Evaluating collaborative filtering model...")
                metrics = app.collab_filter.evaluate_model()
                print("\nCollaborative Filtering Evaluation Results:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")
                    
            elif cmd_type == 'info':
                info = app.get_system_info()
                print("\n" + "="*40)
                print("SYSTEM INFORMATION")
                print("="*40)
                print(f"Total Products: {info['products']:,}")
                print(f"Total Reviews: {info['reviews']:,}")
                print(f"Unique Users: {info['users']:,}")
                print(f"Categories: {info['categories']:,}")
                print(f"Collaborative Filtering: {'Trained' if app.collab_filter.trained else 'Not Trained'}")
                
            elif cmd_type == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

def print_help():
    """Print enhanced help menu with collaborative filtering commands"""
    print("\nAVAILABLE COMMANDS:")
    print("  search best_sellers [category] [n]")
    print("  search rating [operator] [rating] [category] [n]")
    print("  recommend [user_id] [n]          - Hybrid recommendations")
    print("  collab [user_id] [n]             - Collaborative filtering")
    print("  also_bought [product_id] [n]     - Customers also bought")
    print("  similar [product_id] [n]         - Similar products")
    print("  product [product_id]")
    print("  user [user_id]")
    print("  copurchasers [user_id] [product_id]")
    print("  eval_collab                      - Evaluate collaborative model")
    print("  stats")
    print("  info")
    print("  clear")
    print("  help")
    print("  quit/exit")
    print("\nEXAMPLES:")
    print("  search best_sellers Books 10")
    print("  search rating >= 4.5 Books 10")
    print("  recommend CUST_00001 10")
    print("  collab CUST_00001 10")
    print("  also_bought 014241543X 10")
    print("  similar 014241543X 10")
    print("  product 014241543X")
    print("  user CUST_00001")
    print("  copurchasers CUST_00001 014241543X")

def handle_search_command(app, args):
    """Handle enhanced search commands with category and count parameters"""
    if len(args) < 1:
        print("Usage: search [type] [params]")
        print("Types: best_sellers, rating")
        return
        
    query_type = args[0]
    
    try:
        if query_type == 'best_sellers':
            if len(args) >= 2:
                category = args[1]
                n = int(args[2]) if len(args) > 2 else 10
                print(f"Finding top {n} bestsellers in {category}...")
                results = app.execute_search_query('best_sellers', category=category, n=n)
                if results.count() > 0:
                    results.select("product_id", "title", "sales_rank", "average_rating", "review_count", "price").show(n, truncate=50)
                else:
                    print("No results found")
            else:
                print("Usage: search best_sellers [category] [n=10]")
                print("Example: search best_sellers Books 10")
                
        elif query_type == 'rating':
            if len(args) >= 3:
                operator = args[1]
                rating = float(args[2])
                category = args[3] if len(args) > 3 else None
                n = int(args[4]) if len(args) > 4 else 10
                
                print(f"Finding books with rating {operator} {rating} in category {category}...")
                results = app.execute_search_query('rating', operator=operator, rating_threshold=rating, 
                                                 category=category, n=n)
                if results.count() > 0:
                    results.select("product_id", "title", "average_rating", "review_count", "category").show(n, truncate=50)
                else:
                    print("No books found with that rating")
            else:
                print("Usage: search rating [operator] [rating] [category] [n]")
                print("Operators: >, >=, =, <, <=")
                print("Example: search rating >= 4.5 Books 10")
                
        else:
            print(f"Unknown search type: {query_type}")
            print("Available types: best_sellers, rating")
            
    except Exception as e:
        print(f"Search error: {e}")

def run_demo_mode(app):
    """Run demo mode with sample queries including collaborative filtering"""
    print("\n" + "="*50)
    print("AMAZON RECOMMENDER SYSTEM - ENHANCED DEMO MODE")
    print("="*50)
    
    demo_queries = [
        ("System Information", lambda: app.get_system_info()),
        ("Top 10 Bestsellers in Books", lambda: app.execute_search_query('best_sellers', category='Books', n=10)),
        ("Books with High Ratings (>=4.5)", 
         lambda: app.execute_search_query('rating', operator='>=', rating_threshold=4.5, n=10)),
        ("Category Statistics", lambda: app.get_category_stats()),
        ("Hybrid Recommendations for CUST_00001", lambda: app.get_recommendations(user_id='CUST_00001', n=10)),
        ("Collaborative Recommendations for CUST_00001", lambda: app.get_collaborative_recommendations(user_id='CUST_00001', n=10)),
        ("Customers who bought 014241543X also bought", lambda: app.get_also_bought('014241543X', 10)),
        ("Books Similar to 014241543X", lambda: app.get_recommendations(product_id='014241543X', n=10)),
        ("User CUST_00001 Purchase History with Categories", lambda: app.get_user_history('CUST_00001')),
        ("Co-purchasers for User CUST_00001 and Product 014241543X", 
         lambda: print(f"Count: {app.get_copurchasers_count('CUST_00001', '014241543X')}")),
        ("Collaborative Filtering Evaluation", lambda: app.collab_filter.evaluate_model()),
    ]
    
    for description, query_func in demo_queries:
        print(f"\n{description}...")
        try:
            result = query_func()
            if hasattr(result, 'show'):
                result.show(truncate=50)
            elif isinstance(result, dict):
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print(result)
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"Demo query failed: {e}")
    
    print("\nDemo completed! Use --interactive for full control.")

if __name__ == "__main__":
    main()