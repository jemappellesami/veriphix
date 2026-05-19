OPENQASM 2.0;
include "qelib1.inc";
qreg q841[4];
rx(pi/2) q841[0];
cx q841[3],q841[2];
cx q841[0],q841[1];
cx q841[2],q841[1];
rx(pi/4) q841[0];
