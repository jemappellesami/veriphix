OPENQASM 2.0;
include "qelib1.inc";
qreg q476[5];
cx q476[3],q476[4];
cx q476[3],q476[2];
cx q476[2],q476[1];
cx q476[1],q476[0];
rx(pi/4) q476[1];
