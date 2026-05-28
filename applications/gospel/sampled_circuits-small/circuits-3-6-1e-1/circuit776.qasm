OPENQASM 2.0;
include "qelib1.inc";
qreg q777[3];
cx q777[1],q777[0];
rx(pi) q777[1];
cx q777[1],q777[2];
cx q777[0],q777[1];
rx(pi/4) q777[1];
