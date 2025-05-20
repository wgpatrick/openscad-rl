// Simple Hex Nut Model (Unthreaded)
// Author: AI Assistant
// Date: 2023-10-27

// --- Parameters ---

// Flat-to-flat width of the hexagon
hex_width = 10; // [mm]

// Diameter of the central hole
hole_diameter = 5; // [mm]

// Thickness (height) of the nut
nut_thickness = 5; // [mm]


// --- Calculations ---

// Calculate the radius needed for a hexagon with the given flat-to-flat width.
// The radius (vertex to center) r = width / (2 * cos(30))
hex_radius = hex_width / (2 * cos(30));

// Calculate the radius of the hole
hole_radius = hole_diameter / 2;

// Small value to ensure clean cuts in difference operations
epsilon = 0.01;


// --- Model Definition ---

// Use difference() to subtract the hole from the hex prism
difference() {
    // 1. Create the main hexagonal prism body
    linear_extrude(height = nut_thickness, center = true) {
        // Create a 2D hexagon centered at the origin
        // Using circle with $fn=6 creates a hexagon
        // The radius is calculated to match the flat-to-flat width
        circle(r = hex_radius, $fn = 6);
    }

    // 2. Create the cylinder for the central hole
    // Make it slightly taller than the nut thickness for a clean subtraction
    cylinder(h = nut_thickness + 2 * epsilon, r = hole_radius, center = true, $fn=50); // Use more facets for a smoother hole
}

// End of script