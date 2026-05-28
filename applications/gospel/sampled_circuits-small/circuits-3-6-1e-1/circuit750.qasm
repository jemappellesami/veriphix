OPENQASM 2.0;
include "qelib1.inc";
qreg q751[3];
rx(7*pi/4) q751[2];
cx q751[1],q751[2];
cx q751[0],q751[1];
rx(pi/4) q751[1];
