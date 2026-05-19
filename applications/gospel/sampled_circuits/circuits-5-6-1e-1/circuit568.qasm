OPENQASM 2.0;
include "qelib1.inc";
qreg q569[5];
cx q569[3],q569[4];
cx q569[2],q569[3];
cx q569[1],q569[2];
cx q569[0],q569[1];
rx(pi/4) q569[1];
