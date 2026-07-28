"""
Campus Navigation App - Usage Examples
Demonstrates various use cases of the navigation system
"""

import logging
from pathlib import Path
import sys

# Setup paths and logging
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app import CampusNavigationApp
from data_loader import DataLoader


def example_1_basic_location_identification():
    """
    Example 1: Identify location from an image
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Basic Location Identification")
    logger.info("="*70)
    
    app = CampusNavigationApp()
    
    # Find a test image
    test_dir = BASE_DIR / "dataset" / "test"
    if not test_dir.exists():
        logger.warning("Test dataset not found. Skipping example 1.")
        return
    
    # Find first available test image
    for location_dir in sorted(test_dir.iterdir()):
        if location_dir.is_dir():
            images = list(location_dir.glob("*.jpg")) + list(location_dir.glob("*.png"))
            if images:
                test_image = str(images[0])
                logger.info(f"\nTesting with: {Path(test_image).parent.name}/{Path(test_image).name}")
                
                result = app.identify_location(test_image)
                
                logger.info(f"\n✓ Identified Location: {result.get('location', 'Unknown')}")
                logger.info(f"  Confidence: {result.get('confidence', 0):.2%}")
                logger.info(f"  Top 5 matches: {result.get('top_5_matches', [])}")
                logger.info(f"  Distances: {[f'{d:.2f}' for d in result.get('distances', [])]}")
                
                if 'voting_results' in result:
                    logger.info(f"  Voting results: {result['voting_results']}")
                
                return result


def example_2_navigation_path():
    """
    Example 2: Get navigation path between two locations
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Navigation Path")
    logger.info("="*70)
    
    app = CampusNavigationApp()
    
    # Get available locations
    locations = app.get_available_locations()
    logger.info(f"\nAvailable locations: {locations}")
    
    if len(locations) >= 2:
        start = locations[0]
        end = locations[1]
        
        logger.info(f"\nFinding path from '{start}' to '{end}'...")
        
        result = app.get_navigation_path(start, end)
        
        if "error" not in result:
            logger.info(f"\n✓ Path found!")
            logger.info(f"  Route: {' → '.join(result['path'])}")
            logger.info(f"  Distance: {result['total_distance_meters']} meters")
            logger.info(f"  Estimated time: {result['estimated_time_minutes']:.1f} minutes")
            logger.info(f"  Number of stops: {result['num_locations']}")
        else:
            logger.warning(f"  Error: {result['error']}")
        
        return result


def example_3_full_navigation():
    """
    Example 3: Complete navigation query (identify + pathfinding)
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Full Navigation Query")
    logger.info("="*70)
    
    app = CampusNavigationApp()
    
    # Find a test image
    test_dir = BASE_DIR / "dataset" / "test"
    if not test_dir.exists():
        logger.warning("Test dataset not found. Skipping example 3.")
        return
    
    locations = app.get_available_locations()
    if len(locations) < 2:
        logger.warning("Not enough locations for navigation example.")
        return
    
    # Find first test image
    for location_dir in sorted(test_dir.iterdir()):
        if location_dir.is_dir():
            images = list(location_dir.glob("*.jpg")) + list(location_dir.glob("*.png"))
            if images:
                test_image = str(images[0])
                destination = locations[1] if locations[0] != location_dir.name else locations[0]
                
                logger.info(f"\nStarting from: {test_image}")
                logger.info(f"Destination: {destination}")
                
                result = app.full_navigation_query(test_image, destination)
                
                # Display results
                loc_result = result.get('location_identification', {})
                nav_result = result.get('navigation', {})
                
                logger.info(f"\n✓ Current Location: {loc_result.get('location', 'Unknown')}")
                logger.info(f"  Confidence: {loc_result.get('confidence', 0):.2%}")
                
                if 'path' in nav_result:
                    logger.info(f"\n✓ Navigation Path:")
                    logger.info(f"  Route: {' → '.join(nav_result['path'])}")
                    logger.info(f"  Distance: {nav_result['total_distance_meters']} meters")
                    logger.info(f"  Est. time: {nav_result['estimated_time_minutes']:.1f} minutes")
                else:
                    logger.warning(f"  Navigation error: {nav_result.get('error', 'Unknown')}")
                
                return result


def example_4_batch_processing():
    """
    Example 4: Batch process multiple test images
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Batch Processing")
    logger.info("="*70)
    
    app = CampusNavigationApp()
    
    test_dir = BASE_DIR / "dataset" / "test"
    if not test_dir.exists():
        logger.warning("Test dataset not found. Skipping example 4.")
        return
    
    logger.info(f"\nProcessing test images...")
    
    results_summary = {}
    
    for location_dir in sorted(test_dir.iterdir()):
        if location_dir.is_dir():
            images = list(location_dir.glob("*.jpg")) + list(location_dir.glob("*.png"))
            
            if images:
                location_name = location_dir.name
                logger.info(f"\n  Location: {location_name}")
                
                correct = 0
                total = min(5, len(images))  # Process max 5 images per location
                
                for image_path in images[:total]:
                    result = app.identify_location(str(image_path))
                    identified = result.get('location', 'Unknown')
                    
                    is_correct = identified == location_name
                    if is_correct:
                        correct += 1
                    
                    status = "✓" if is_correct else "✗"
                    logger.info(f"    {status} {image_path.name}: {identified}")
                
                accuracy = (correct / total * 100) if total > 0 else 0
                logger.info(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
                
                results_summary[location_name] = accuracy
    
    logger.info(f"\n✓ Batch Processing Summary:")
    overall_accuracy = sum(results_summary.values()) / len(results_summary) if results_summary else 0
    for location, acc in sorted(results_summary.items()):
        logger.info(f"  {location}: {acc:.1f}%")
    logger.info(f"  Overall: {overall_accuracy:.1f}%")
    
    return results_summary


def example_5_index_statistics():
    """
    Example 5: Display index statistics
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: Index Statistics")
    logger.info("="*70)
    
    app = CampusNavigationApp()
    stats = app.get_index_statistics()
    
    logger.info(f"\n✓ Index Information:")
    logger.info(f"  Total indexed images: {stats['total_indexed_items']}")
    logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")
    logger.info(f"  Number of locations: {len(stats['locations'])}")
    
    logger.info(f"\n✓ Images per location:")
    for location, count in sorted(stats['items_per_location'].items()):
        logger.info(f"  {location}: {count} images")


def example_6_alternative_routes():
    """
    Example 6: Find alternative routes
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 6: Alternative Routes")
    logger.info("="*70)
    
    app = CampusNavigationApp()
    locations = app.get_available_locations()
    
    if len(locations) >= 2:
        start = locations[0]
        end = locations[-1]
        
        logger.info(f"\nFinding routes from '{start}' to '{end}'...")
        
        # Find main path
        main_path = app.get_navigation_path(start, end)
        
        if 'path' in main_path:
            logger.info(f"\n✓ Optimal Route:")
            logger.info(f"  Path: {' → '.join(main_path['path'])}")
            logger.info(f"  Distance: {main_path['total_distance_meters']} meters")
            logger.info(f"  Time: {main_path['estimated_time_minutes']:.1f} minutes")
        
        logger.info(f"\nNote: For alternative routes, modify the campus graph in pathfinding.py")


def example_7_custom_graph():
    """
    Example 7: Using custom campus graph
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 7: Custom Campus Graph")
    logger.info("="*70)
    
    from pathfinding import CampusGraph, AStarPathfinder
    
    # Create custom graph
    graph = CampusGraph()
    
    logger.info(f"\n✓ Default campus locations: {list(graph.graph.keys())}")
    
    # Example of adding a new location
    logger.info(f"\n✓ Adding new location 'Gym'...")
    graph.add_edge("Cafe", "Gym", distance=250)
    graph.add_edge("Gym", "Courtyard", distance=180)
    graph.set_coordinates("Gym", x=50, y=-100)
    
    # Find path through new location
    pathfinder = AStarPathfinder(graph)
    path = pathfinder.find_path("Cafe", "Gym")
    
    if path:
        logger.info(f"  Path from Cafe to Gym: {' → '.join(path)}")
        time_estimate = pathfinder.estimate_travel_time(path)
        logger.info(f"  Estimated time: {time_estimate/60:.1f} minutes")


def main():
    """Run all examples"""
    logger.info("\n" + "="*70)
    logger.info("CAMPUS NAVIGATION APP - USAGE EXAMPLES")
    logger.info("="*70)
    
    try:
        # Run examples
        example_1_basic_location_identification()
        example_2_navigation_path()
        example_3_full_navigation()
        example_4_batch_processing()
        example_5_index_statistics()
        example_6_alternative_routes()
        example_7_custom_graph()
        
        logger.info("\n" + "="*70)
        logger.info("✓ ALL EXAMPLES COMPLETED")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\nError running examples: {e}", exc_info=True)
        logger.info("\nMake sure to run 'python train_index.py' first to build the index!")


if __name__ == "__main__":
    main()
