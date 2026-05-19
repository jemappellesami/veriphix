OPENQASM 2.0;
include "qelib1.inc";
qreg q132[5];
cx q132[3],q132[4];
cx q132[2],q132[3];
cx q132[1],q132[2];
cx q132[1],q132[0];
rx(pi/4) q132[1];
