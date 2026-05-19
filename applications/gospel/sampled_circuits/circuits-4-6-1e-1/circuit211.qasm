OPENQASM 2.0;
include "qelib1.inc";
qreg q212[4];
rx(7*pi/4) q212[3];
cx q212[3],q212[2];
cx q212[1],q212[2];
cx q212[0],q212[1];
rx(pi/4) q212[1];
