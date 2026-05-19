OPENQASM 2.0;
include "qelib1.inc";
qreg q579[4];
cx q579[1],q579[0];
cx q579[0],q579[1];
cx q579[1],q579[0];
cx q579[1],q579[2];
cx q579[1],q579[0];
