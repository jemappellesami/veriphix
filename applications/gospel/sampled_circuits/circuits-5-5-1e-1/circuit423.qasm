OPENQASM 2.0;
include "qelib1.inc";
qreg q424[5];
rz(5*pi/4) q424[3];
rx(pi/2) q424[3];
cx q424[2],q424[3];
cx q424[2],q424[1];
cx q424[1],q424[0];
