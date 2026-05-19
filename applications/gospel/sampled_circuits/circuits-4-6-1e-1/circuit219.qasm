OPENQASM 2.0;
include "qelib1.inc";
qreg q220[4];
cx q220[2],q220[3];
cx q220[3],q220[2];
cx q220[1],q220[2];
cx q220[0],q220[1];
rx(pi/4) q220[1];
