// Simple Hex Nut Model (No Threads)
//
// This script creates a single, manifold 3D model of a hex nut
// centered at the origin (0,0,0).

// --- Parameters ---

// Flat-to-flat width of the hexagon (mm)
nut_width = 10;

// Diameter of the central hole (mm)
hole_diameter = 5;

// Thickness (height) of the nut (mm)
nut_thickness = 5;

// --- Calculations ---

// OpenSCAD's cylinder function uses radius (center-to-vertex for polygons).
// We need to calculate this radius from the flat-to-flat width.
// Formula: width = 2 * radius * cos(30 degrees)
// Therefore: radius = width / (2 * cos(30))
hex_radius = nut_width / (2 * cos(30));

// --- Model Definition ---

// Use difference() to subtract the hole from the hexagonal prism.
difference() {
    // 1. Create the main hexagonal prism body.
    // Use cylinder() with $fn=6 to create a hexagon.
    // Center=true ensures it's centered vertically along the Z-axis.
    cylinder(h = nut_thickness, r = hex_radius, $fn = 6, center = true);

    // 2. Create the central hole cylinder to be subtracted.
    // Make the hole cylinder slightly taller than the nut thickness (epsilon)
    // to ensure a clean subtraction and avoid potential coincident face issues.
    // Center=true aligns it with the hexagonal prism.
    cylinder(h = nut_thickness + 0.02, d = hole_diameter, $fn = 50, center = true); // Use $fn for smoother hole
}

// End of script