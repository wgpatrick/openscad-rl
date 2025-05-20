// OpenSCAD script for a milled plate model

// Main model: start with a 45x60x10mm plate and subtract the holes, pocket, slot, and center through‐hole
difference() {
  
  // 1. Create the main plate (cube from (0,0,0) to (45,60,10))
  cube([45,60,10]);
  
  // 2. Subtract 4 corner holes (3mm diameter, full-through holes)
// Each hole is centered 5mm from the adjacent edges.
// The cylinder is given extra height (12mm) and lowered slightly in Z to ensure it cuts through.
  translate([5,5,-1])
    cylinder(d=3, h=12, $fn=50);
    
  translate([5,55,-1])
    cylinder(d=3, h=12, $fn=50);
    
  translate([40,5,-1])
    cylinder(d=3, h=12, $fn=50);
    
  translate([40,55,-1])
    cylinder(d=3, h=12, $fn=50);
  
  // 3. Subtract the rectangular pocket with rounded corners
// Pocket dimensions: 22.5mm x 30mm in XY with 5mm radius corners,
// cut from the top surface down 6mm (leaving a 4mm bottom).
// The pocket is exactly centered in the plate.
// Build the 2D pocket shape using minkowski sum (a square plus a circle for fillets)
// The inner square is chosen with dimensions reduced by twice the fillet radius.
  translate([0,0,4])  // Position the pocket so that it spans z from 4mm to 10mm
    linear_extrude(height=6)
      // Translate the 2D pocket shape to its XY location (centered in the 45x60 plate)
      translate([11.25,15])
        minkowski() {
          // The inner square: dimensions chosen so that after adding fillets the overall
          // width becomes 12.5+10 = 22.5mm and height 20+10 = 30mm.
          square([12.5,20], center=false);
          // Circle for rounded corners with radius 5mm
          circle(r=5, $fn=50);
        }
        
  // 4. Subtract the slot from the outside of the plate to the pocket
// The slot is 6mm wide (in Y), parallel to X axis, with a depth of 6mm (z from 4mm to 10mm).
// It connects the left edge of the plate (x=0) to the left side of the pocket (x=11.25).
// The slot is centered vertically (Y center = 30, so Y from 27 to 33).
  translate([0,27,4])
    cube([11.25,6,6]);
    
  // 5. Subtract a 15mm diameter through hole at the center of the plate.
// The center of the plate is at (22.5,30) in XY. The cylinder extends sufficiently in Z.
  translate([22.5,30,-1])
    cylinder(d=15, h=12, $fn=100);
}