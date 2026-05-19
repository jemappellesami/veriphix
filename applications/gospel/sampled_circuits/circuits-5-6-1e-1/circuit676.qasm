OPENQASM 2.0;
include "qelib1.inc";
qreg q677[5];
cx q677[3],q677[4];
cx q677[2],q677[3];
cx q677[2],q677[1];
cx q677[1],q677[0];
rx(pi/4) q677[1];
