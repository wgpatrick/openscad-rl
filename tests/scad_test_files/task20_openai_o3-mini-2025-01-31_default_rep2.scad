// OpenSCAD script for a single-part milled workpiece
// The part is defined with the lower‐left corner at (0,0,0)
// Dimensions: 80mm (X) x 50mm (Y) x 20mm (Z)

// ////////////////////////////////////////////////////////////////////
// Helper module: Creates a 2D rounded rectangle using a Minkowski sum.
// This produces a shape with overall width = w and height = h,
// with filleted (rounded) corners of radius r.
// The "center" flag sets the polygon’s center to (0,0) when true.
module rounded_rect(w, h, r, center=false) {
    if (center)
        translate([ - (w-2*r)/2, - (h-2*r)/2 ])
            minkowski() {
                square([w-2*r, h-2*r], center=false);
                circle(r, $fn=50);
            }
    else
        minkowski() {
            square([w-2*r, h-2*r], center=false);
            circle(r, $fn=50);
        }
}

// ////////////////////////////////////////////////////////////////////
// Step 1 & 2: Create the main body.
// Instead of filleting edges after extruding a cube,
// we build the 2D profile as a rounded rectangle and extrude it.
// The main 2D profile: 80 x 50 with corner fillet radius 15.
main_body = linear_extrude(height = 20)
    rounded_rect(80, 50, 15, center = false);

// ////////////////////////////////////////////////////////////////////
// Step 3: Create a cut feature that “mills” down 15mm along the perimeter.
// The idea is to remove a ring defined by the difference of the
// original profile and that profile offset inward by 5mm.
// For a constant offset of 5mm, the inner profile dimensions are:
//   Width: 80 - 2*5 = 70
//   Height: 50 - 2*5 = 40
// and the fillet radius reduces by 5 to become 15 - 5 = 10.
// We extrude this ring shape 15mm in Z. Because the cut starts
// at the top (Z = 20) and goes down 15mm, we translate it so that
// its bottom is at Z = 5.
ring_cut = translate([0, 0, 5])
    linear_extrude(height = 15)
        difference() {
            // Outer profile (same as the main body profile)
            rounded_rect(80, 50, 15, center = false);
            // Inner profile, offset inwards by 5 mm on all sides.
            rounded_rect(70, 40, 10, center = false);
        };

// ////////////////////////////////////////////////////////////////////
// Step 4: Create the through hole.
// A cylinder (hole) of diameter 5mm (radius 2.5mm) is cut through the part.
// Its center is positioned 6mm from the midline (in Y) and 10mm from the edge (in X).
// Since the main body extends from X = 0 to 80 and Y = 0 to 50,
// the midline in Y is at 25mm and 6mm above that gives Y = 31mm,
// and “10mm from the edge” (choosing the left edge) places the center at X = 10.
// The cylinder is made tall enough to pass completely through (using extra height).
hole_cut = translate([10, 31, -5])
    cylinder(h = 30, r = 2.5, center = false, $fn = 50);

// ////////////////////////////////////////////////////////////////////
// Step 5: Create the large rectangular slot cut.
// The slot is a 2D rounded rectangle with overall dimensions 50mm (X) x 20mm (Y)
// and filleted corners of radius 5mm. It is centered in the XY plane of the part.
// The center of the overall part is at (40,25) (since the main body starts at (0,0)).
// We generate a centered 2D slot shape and extrude it through the entire 20mm.
slot_cut = translate([40, 25, 0])
    linear_extrude(height = 20)
        // With center=true, the shape is symmetric around (0,0) and then moved.
        rounded_rect(50, 20, 5, center = true);

// ////////////////////////////////////////////////////////////////////
// Final model: Subtract the cut features (ring_cut, hole_cut, slot_cut)
// from the main body.
difference() {
    // Main solid body with rounded (filleted) edges.
    main_body;
    // Milled downward cut along the perimeter.
    ring_cut;
    // Through hole.
    hole_cut;
    // Rectangular slot.
    slot_cut;
}