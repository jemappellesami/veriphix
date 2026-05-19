OPENQASM 2.0;
include "qelib1.inc";
qreg q514[4];
rx(7*pi/4) q514[0];
cx q514[0],q514[1];
cx q514[2],q514[1];
rx(pi/4) q514[0];
