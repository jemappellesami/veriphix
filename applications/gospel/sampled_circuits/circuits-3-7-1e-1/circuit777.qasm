OPENQASM 2.0;
include "qelib1.inc";
qreg q778[3];
cx q778[2],q778[1];
rz(3*pi/2) q778[2];
cx q778[1],q778[2];
cx q778[1],q778[0];
