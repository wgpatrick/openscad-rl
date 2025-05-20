// Simple Hex Nut Model (No Threads)
// Author: AI Assistant
// Date: 2023-10-27

// --- Parameters ---

// Width of the hexagon measured between two parallel flat sides (mm)
hex_flat_to_flat_width = 10;

// Diameter of the central hole (mm)
hole_diameter = 5;

// Thickness (height) of the nut (mm)
nut_thickness = 5;


// --- Derived Variables ---

// Calculate the radius needed for a circle primitive to create a hexagon
// with the specified flat-to-flat width.
// The radius of the circumscribed circle of a hexagon is r = width / (2 * cos(30))
hex_radius = hex_flat_to_flat_width / (2 * cos(30));

// Calculate the radius of the central hole
hole_radius = hole_diameter / 2;


// --- Model Definition ---

// Use the difference() operation to subtract the hole from the hexagonal prism.
difference() {
    // 1. Create the outer hexagonal prism
    // Extrude a 2D hexagon along the Z-axis.
    // Center=true ensures the model is centered vertically around Z=0.
    linear_extrude(height = nut_thickness, center = true) {
        // Create a 2D hexagon centered at the origin (0,0).
        // Use circle() with $fn=6 to define the hexagon.
        // The radius is calculated to match the desired flat-to-flat width.
        circle(r = hex_radius, $fn = 6);
    }

    // 2. Create the cylinder for the central hole to be subtracted
    // Make the cylinder slightly taller than the nut thickness to ensure a clean boolean subtraction.
    // Center=true aligns it vertically with the hexagonal prism.
    linear_extrude(height = nut_thickness + 2, center = true) { // Add small epsilon (2mm total) for clean cut
        // Create a 2D circle for the hole, centered at the origin (0,0).
        // Use the calculated hole radius. Use default $fn for a smooth cylinder.
        circle(r = hole_radius);
    }
}

// End of script