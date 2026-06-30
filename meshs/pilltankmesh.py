# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:12:36 2026

@author: Owner
"""

import argparse
import trimesh

def create_capsule_tank_mesh(radius, height, resolution):
    """
    Generates a pill-shaped tank (capsule) CAD mesh.
    radius: Radius of the cylinder and hemisphere caps
    height: Height of the central cylindrical section
    resolution: Tuple of [latitudes, longitudes] mesh resolution
    """
    # Trimesh automatically handles watertight stitching of the cylinder and caps
    capsule = trimesh.creation.capsule(height=height, radius=radius, count=resolution)
    return capsule

def main():
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Generate a 3D CAD mesh of a pill-shaped tank (capsule)."
    )
    
    # Required arguments
    parser.add_argument(
        '-r', '--radius', 
        type=float, 
        required=True, 
        help="Radius of the cylinder and hemisphere caps (e.g., 1.5)"
    )
    parser.add_argument(
        '-H', '--height', 
        type=float, 
        required=True, 
        help="Height of the central cylindrical section (excluding the caps)"
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='pill_shaped_tank', 
        help="Base name for the output files (default: 'pill_shaped_tank')"
    )
    parser.add_argument(
        '--res', 
        type=int, 
        nargs=2, 
        default=[20, 20], 
        metavar=('LAT', 'LONG'),
        help="Mesh resolution as [latitudes, longitudes] (default: 20 20)"
    )

    # Parse arguments
    args = parser.parse_args()

    print(f"Generating tank mesh: Radius = {args.radius}, Central Height = {args.height}...")
    print(f"Total end-to-end tank length will be: {args.height + (2 * args.radius)}")

    # Generate the mesh
    tank_mesh = create_capsule_tank_mesh(
        radius=args.radius, 
        height=args.height, 
        resolution=args.res
    )

    # Define export filenames
    stl_filename = f"{args.output}.stl"
    obj_filename = f"{args.output}.obj"

    # Export to standard CAD mesh formats
    tank_mesh.export(stl_filename)
    tank_mesh.export(obj_filename)

    print(f"\nSaved meshes to:")
    print(f"  - {stl_filename}")
    print(f"  - {obj_filename}")
    print(f"Mesh summary: {len(tank_mesh.vertices)} vertices, {len(tank_mesh.faces)} faces.")

if __name__ == '__main__':
    main()