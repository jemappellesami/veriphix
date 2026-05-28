OPENQASM 2.0;
include "qelib1.inc";
qreg q966[3];
rz(3*pi/4) q966[2];
cx q966[2],q966[1];
cx q966[1],q966[0];
