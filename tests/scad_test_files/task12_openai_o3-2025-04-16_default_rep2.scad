/* 
  T‑Handle Tap Holder 
  All dimensions are in millimetres.
  
  Geometry summary
  ----------------
  1. Shank      : Ø15 × 60 (Z) with a 1 mm 45° chamfer on the end that has the tap hole.
  2. Tap hole   : Ø8 × 25 deep, concentric with the shank, starting at the chamfered end.
  3. T‑handle   : Ø5 × 60, orthogonal to the shank (along X), centred 5.5 mm up from the opposite end.
  
  The resulting model is a single, watertight, manifold solid.
*/

$fn = 100;                         // Resolution for circular features

//---------------------------------
// Helper ‑ 45° chamfered cylinder
//---------------------------------
module chamfered_cylinder(r=7.5, h=60, chamfer=1)
{
    /*  Two coaxial cylinders are hulled together:
        – a full‑diameter body of height h‑chamfer
        – a reduced‑diameter top section of height chamfer
        The hull naturally forms the 45° chamfer.          */
    hull() {
        cylinder(r = r,           h = h - chamfer);
        translate([0, 0, h - chamfer])
            cylinder(r = r - chamfer, h = chamfer);
    }
}

//--------------------
// Core model
//--------------------
difference() {

    //--- SOLID: shank + T‑handle (union) ------------------
    union() {
        // Shank with 1 mm chamfer on the “tap” end (Z = 60)
        chamfered_cylinder(r = 7.5, h = 60, chamfer = 1);

        // T‑handle (Ø5 × 60) centred 5.5 mm from the opposite end (Z = 0)
        translate([0, 0, 5.5])
            rotate([0, 90, 0])      // orient along X axis
                cylinder(r = 2.5, h = 60, center = true);
    }

    //--- SUBTRACT: concentric tap hole --------------------
    /*  Extra 0.2 mm ensures a clean cut past the outer face */
    translate([0, 0, 60 - 25])
        cylinder(r = 4, h = 25 + 0.2);
}