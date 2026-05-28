OPENQASM 2.0;
include "qelib1.inc";
qreg q256[3];
rx(pi/2) q256[0];
cx q256[0],q256[1];
cx q256[2],q256[1];
rx(pi/4) q256[0];
