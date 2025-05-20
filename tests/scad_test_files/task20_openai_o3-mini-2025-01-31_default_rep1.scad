// Set resolution for circles and arcs
$fn = 50;

// Base 2D rectangle used for the main shape before rounding.
// When offset with a 15mm radius, it will produce a rounded rectangle with overall dimensions 80 x 50.
module base_rect() {
    // Place a 50mm x 20mm rectangle at (15,15) so that after a 15mm offset,
    // the rounded rectangle spans from (0,0) to (80,50).
    translate([15,15])
        square([50,20], center = false);
}

// Main 2D outline: a rounded rectangle (with a constant 15mm fillet on vertical edges)
module main_outline_2d() {
    offset(r=15, join_type="round")
        base_rect();
}

// Groove profile: the peripheral "ring" formed by the region between the main outline
// and an inner profile offset 5mm toward the center.
module groove_profile_2d() {
    difference() {
        // Outer boundary is the full main outline.
        main_outline_2d();
        // Inner boundary: contract the main outline by 5mm.
        offset(delta = -5, join_type = "round")
            main_outline_2d();
    }
}

// Slot profile: a rounded rectangle of size 50x20 with 5mm filleted corners.
// Constructed by taking a rectangle of size (50 - 2*5) x (20 - 2*5) = 40 x 10 (centered)
// and using a Minkowski sum with a circle of radius 5.
module slot_profile_2d() {
    minkowski() {
        square([40,10], center=true);
        circle(r=5);
    }
}

// Final model
// Start with the main body, then subtract the groove cut, a through hole, and the slot.
difference() {
    // Main body: extrude the rounded rectangle to 20mm in the Z direction.
    linear_extrude(height = 20)
        main_outline_2d();
    
    // Groove cut: remove a 15mm deep peripheral groove.
    // Positioned so that it removes material from z = 5mm to z = 20mm.
    translate([0,0,5])
        linear_extrude(height = 15)
            groove_profile_2d();
    
    // Through hole: a cylinder (5mm diameter) cut completely through the part.
    // The hole is positioned 10mm from the left edge (i.e. x = 10)
    // and 6mm above the horizontal midline (the midline in Y is at 25mm, so y = 31).
    translate([10,31,0])
        cylinder(d = 5, h = 22, center = false);
        // h = 22 to ensure it fully penetrates the 20mm thick part.
    
    // Rectangular slot: a through slot centered in the XY plane.
    // The slot is 50mm (X) x 20mm (Y) with 5mm filleted corners.
    // Since the part spans 80 x 50, its center is at (40,25).
    translate([40,25,0])
        linear_extrude(height = 20)
            slot_profile_2d();
}