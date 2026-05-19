OPENQASM 2.0;
include "qelib1.inc";
qreg q261[3];
rx(3*pi/2) q261[0];
cx q261[0],q261[1];
cx q261[2],q261[1];
cx q261[1],q261[0];
rx(pi/4) q261[1];
