OPENQASM 2.0;
include "qelib1.inc";
qreg q623[3];
cx q623[2],q623[1];
cx q623[1],q623[2];
cx q623[0],q623[1];
