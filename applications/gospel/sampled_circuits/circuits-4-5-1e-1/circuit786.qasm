OPENQASM 2.0;
include "qelib1.inc";
qreg q787[4];
rx(pi) q787[3];
cx q787[3],q787[2];
cx q787[2],q787[1];
cx q787[1],q787[0];
