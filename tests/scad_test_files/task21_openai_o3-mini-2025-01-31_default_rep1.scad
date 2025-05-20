// OpenSCAD script for a milled plate with holes, a pocket, a slot, and a center through‐hole
// Plate dimensions: 45mm (X) x 60mm (Y) x 10mm (Z)
// All features are defined relative to the origin at (0,0,0)

// The model is built as a single part using successive difference() operations

difference() {
  // 1. Base plate: a rectangular prism from (0,0,0)
  cube([45, 60, 10]);
  
  // 2. Four corner through-holes (3mm diameter).
  // Each hole's center is 5mm from each edge.
  // Using cylinders that extend beyond the plate (height=20, centered at z=5) for a full cut.
  translate([5, 5, 5])
    cylinder(d=3, h=20, center=true);
    
  translate([40, 5, 5])
    cylinder(d=3, h=20, center=true);
    
  translate([5, 55, 5])
    cylinder(d=3, h=20, center=true);
    
  translate([40, 55, 5])
    cylinder(d=3, h=20, center=true);
  
  // 3. Rectangular pocket with rounded corners.
  // Pocket dimensions: 22.5mm (X) x 30mm (Y), with a corner radius of 5mm.
  // The pocket is cut 6mm deep from the top surface (i.e. from z=4 to z=10).
  // The pocket is centered on the plate.
  translate([22.5, 30, 4])
    linear_extrude(height=6)
      // Create a round-cornered shape by starting with a smaller centered square
      // and offsetting it outward by 5mm.
      offset(r=5)
        square([12.5, 20], center=true);
  
  // 4. Slot from the outside of the plate to the rectangular pocket.
  // The slot is 6mm wide (Y), 6mm deep (Z, from z=4 to z=10), and runs parallel to the X axis.
  // It is centered in the Y direction (centered at y=30).
  // Assuming the slot starts at the left edge (x=0) and extends to the left side of the pocket (x=11.25).
  translate([0, 27, 4])
    cube([11.25, 6, 6]);
  
  // 5. Through-hole in the center of the plate.
  // Hole diameter: 15mm. The hole passes completely through the plate.
  // Using a cylinder (height=20, centered at z=5) to ensure a complete cut.
  translate([22.5, 30, 5])
    cylinder(d=15, h=20, center=true);
}