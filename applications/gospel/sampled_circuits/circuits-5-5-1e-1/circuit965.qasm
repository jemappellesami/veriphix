OPENQASM 2.0;
include "qelib1.inc";
qreg q966[5];
rz(5*pi/4) q966[1];
cx q966[2],q966[3];
cx q966[1],q966[0];
cx q966[2],q966[1];
rx(pi/4) q966[0];
