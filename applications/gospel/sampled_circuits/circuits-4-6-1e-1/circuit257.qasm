OPENQASM 2.0;
include "qelib1.inc";
qreg q258[4];
cx q258[2],q258[1];
cx q258[3],q258[2];
cx q258[2],q258[1];
cx q258[0],q258[1];
rx(pi/4) q258[1];
