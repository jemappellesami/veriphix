OPENQASM 2.0;
include "qelib1.inc";
qreg q78[3];
cx q78[1],q78[0];
cx q78[0],q78[1];
rx(pi) q78[1];
cx q78[1],q78[0];
cx q78[2],q78[1];
