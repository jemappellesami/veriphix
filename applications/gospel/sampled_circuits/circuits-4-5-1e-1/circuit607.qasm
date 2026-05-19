OPENQASM 2.0;
include "qelib1.inc";
qreg q608[4];
cx q608[2],q608[1];
cx q608[1],q608[2];
cx q608[1],q608[0];
cx q608[2],q608[3];
