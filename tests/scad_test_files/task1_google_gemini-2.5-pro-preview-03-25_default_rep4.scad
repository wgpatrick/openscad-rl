// Simple Hex Nut Model
// Author: AI Assistant
// Date: 2023-10-27

// --- Parameters ---

// Width of the hexagon measured between two parallel flat sides
hex_flat_to_flat_width = 10; // [mm]

// Diameter of the central hole
hole_diameter = 5; // [mm]

// Thickness of the nut (height of the extrusion)
nut_thickness = 5; // [mm]

// --- Calculations ---

// Calculate the radius of the hexagon's circumscribed circle (distance from center to vertex)
// R = (W/2) / cos(30), where W is the flat-to-flat width
hex_radius = (hex_flat_to_flat_width / 2) / cos(30);

// Define a small value to ensure the hole cuts completely through
epsilon = 0.1; // [mm]

// --- Model Definition ---

// Use difference() to subtract the hole from the hexagonal prism
difference() {
    // 1. Create the outer hexagonal prism
    linear_extrude(height = nut_thickness, center = true) {
        // Create a 2D hexagon centered at the origin
        // Use circle() with $fn=6 for a regular hexagon
        // The radius 'r' corresponds to the distance from the center to a vertex
        circle(r = hex_radius, $fn = 6);
    }

    // 2. Create the central cylindrical hole
    // Make the cylinder slightly taller than the nut thickness to ensure a clean cut
    cylinder(d = hole_diameter, h = nut_thickness + 2 * epsilon, center = true);
}

// End of script