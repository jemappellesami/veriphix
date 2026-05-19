OPENQASM 2.0;
include "qelib1.inc";
qreg q130[5];
rx(3*pi/4) q130[0];
cx q130[4],q130[3];
cx q130[3],q130[2];
cx q130[1],q130[2];
cx q130[1],q130[0];
