OPENQASM 2.0;
include "qelib1.inc";
qreg q777[6];
cx q777[3],q777[4];
cx q777[3],q777[2];
cx q777[5],q777[4];
cx q777[2],q777[1];
cx q777[0],q777[1];
