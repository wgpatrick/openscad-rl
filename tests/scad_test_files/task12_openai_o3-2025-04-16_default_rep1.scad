//--------------------------------------------------------------
// T‑handle tap holder
// (single, manifold part as specified)
//--------------------------------------------------------------

// --- Global quality setting ----------------------------------
$fn = 100;                 // smoothness for circular features

//--------------------------------------------------------------
// Build the part
//--------------------------------------------------------------
difference() {

    //---------------  SOLID GEOMETRY  -------------------------
    union() {

        //------------------- Shank (Ø15 × 60) with 1 mm chamfer
        //
        //  • Main body: full‑radius cylinder, 59 mm long
        //  • Chamfer: 1 mm high 45° frustum
        //
        translate([0,0, 1])               // start main body at Z = 1 mm
            cylinder(d = 15, h = 59);     // ends at Z = 60 mm

        cylinder(h = 1,                   // chamfer section (Z = 0 → 1 mm)
                 r1 = 7.5,                // radius at Z = 0 mm
                 r2 = 6.5);               // radius at Z = 1 mm


        //------------------- T‑handle (Ø5 × 60, centred) -------
        //
        //  • Axis along X, total length 60 mm (±30 mm from origin)
        //  • Centre positioned at Z = 5.5 mm
        //
        translate([0, 0, 5.5])            // locate handle along Z
            rotate([0, 90, 0])            // align cylinder along X
                cylinder(d = 5, h = 60, center = true);
    }

    //-------------------- Axial tap hole (Ø8 × 25) -------------
    translate([0, 0, 0])                  // starts flush with shank end
        cylinder(d = 8, h = 25);          // depth 25 mm into shank
}