OPENQASM 2.0;
include "qelib1.inc";
qreg q759[5];
rx(pi/2) q759[0];
cx q759[3],q759[4];
cx q759[2],q759[3];
cx q759[1],q759[2];
cx q759[0],q759[1];
