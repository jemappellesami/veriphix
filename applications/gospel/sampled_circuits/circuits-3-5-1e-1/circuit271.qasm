OPENQASM 2.0;
include "qelib1.inc";
qreg q272[3];
rx(5*pi/4) q272[0];
cx q272[0],q272[1];
cx q272[1],q272[2];
cx q272[0],q272[1];
