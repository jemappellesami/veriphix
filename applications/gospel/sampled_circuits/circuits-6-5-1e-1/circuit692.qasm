OPENQASM 2.0;
include "qelib1.inc";
qreg q693[6];
cx q693[5],q693[4];
cx q693[3],q693[4];
cx q693[3],q693[2];
cx q693[1],q693[2];
cx q693[1],q693[0];
